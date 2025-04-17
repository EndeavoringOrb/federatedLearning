import socket
import threading
import queue
import concurrent.futures
from time import sleep, perf_counter
import json
import os
import numpy as np

from utilities.communication import (
    CommunicationHandler,
    log,
)
from utilities.model import *
from shakespeareData.shakespeareData import tokenLoader


# Thread-safe client management using a queue-based approach
class ClientManager:
    def __init__(self):
        self.active_clients = []
        self.new_clients_queue = queue.Queue()
        self.lock = threading.RLock()  # For active_clients only

    def add_new_client(self, client_socket, client_addr):
        """Add a new client to the queue - no locking needed"""
        self.new_clients_queue.put((client_socket, client_addr))

    def get_new_clients(self):
        """Get all new clients without blocking - no locking needed"""
        new_clients = []
        while not self.new_clients_queue.empty():
            try:
                new_clients.append(self.new_clients_queue.get_nowait())
            except queue.Empty:
                break
        return new_clients

    def activate_client(self, client):
        """Move a client to active status"""
        with self.lock:
            self.active_clients.append(client)

    def get_active_clients(self):
        """Get a copy of active clients list"""
        with self.lock:
            return self.active_clients.copy()

    def remove_client(self, client):
        """Remove a client from active list"""
        client_socket, client_addr = client
        try:
            log.warning(f"Removing client {client_addr}.")
            client_socket.close()
        except Exception as e:
            log.error(f"Error closing socket for {client_addr}: {e}")

        with self.lock:
            if client in self.active_clients:
                self.active_clients.remove(client)

    def count_clients(self):
        """Get total count of clients"""
        with self.lock:
            active_count = len(self.active_clients)
        return active_count + self.new_clients_queue.qsize()


class TrainingServer:
    def __init__(self):
        # Configuration
        self.config = {
            "timePerStep": 10,
            "learningRate": 1e-3,
            "sigma": 1e-2,
            "hiddenSize": 16,
            "vocabSize": 76,
            "nLayers": 4,
            "optimizer": "adam",  # "sgd" or "adam"
            "beta1": 0.9,
            "beta2": 0.999,
            "stepNum": 0,
            "checkPointTime": 60,
            "newTokensInterval": 1e9,  # send new tokens every N steps
            "batchSize": 1,
            "modelType": "chat",
        }
        self.seed_high = 4_000_000

        # Initialize model and tokens
        self.tokens = tokenLoader(self.config["vocabSize"], self.config["batchSize"])
        self.model = ChatModel()
        self.weights = self.model.getWeights(
            self.config["hiddenSize"], self.config["vocabSize"], self.config["nLayers"]
        )
        self.n_params = self.weights.shape[0]
        log.info(f"Model has {self.n_params:,} parameters")

        self.optimizer_values = np.zeros(self.n_params * 2).astype(np.float32)
        self.step_num = 0
        self.last_checkpoint_time = perf_counter()
        self.all_rewards = []
        self.reward_info = []

        # Initialize client manager
        self.client_manager = ClientManager()

        # Control flow
        self.running = False
        self.current_training_run = self._setup_training_run()

        # Thread pool for handling client communications
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def _setup_training_run(self):
        """Setup the training run directory and return the run number"""
        if not os.path.exists("trainingRuns"):
            os.makedirs("trainingRuns")

        if len(os.listdir("trainingRuns")) == 0:
            current_run = 0
        else:
            # Find the highest numbered directory
            run_dirs = [d for d in os.listdir("trainingRuns") if d.isdigit()]
            current_run = max([int(item) for item in run_dirs]) + 1 if run_dirs else 0

        run_path = f"trainingRuns/{current_run}"
        os.makedirs(run_path, exist_ok=True)
        log.info(f"Starting training run {current_run}")

        # Save config
        config_path = f"{run_path}/config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
            log.info(f"Saved config to {config_path}")

        return current_run

    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = f"trainingRuns/{self.current_training_run}"
        os.makedirs(checkpoint_path, exist_ok=True)

        model_data = np.concatenate(
            (
                [
                    self.config["hiddenSize"],
                    self.config["vocabSize"],
                    self.config["nLayers"],
                ],
                self.weights,
            )
        )

        np.save(f"{checkpoint_path}/model.npy", model_data)
        log.info(f"Checkpoint saved at step {self.step_num}.")
        self.last_checkpoint_time = perf_counter()

    def _get_weights_from_client(self, client):
        """Get weights from a client"""
        client_socket, client_addr = client

        try:
            # Request weights
            CommunicationHandler.send_message(client_socket, "need weights")

            # Receive weights data
            data = CommunicationHandler.receive_message(client_socket)

            log.info(f"[{client_addr}] Sent weights data.")
            self.config["stepNum"] = int(data[0])
            self.step_num = self.config["stepNum"]
            self.optimizer_values = data[1 : 1 + 2 * self.n_params]
            self.weights = data[1 + 2 * self.n_params :]

            self.save_checkpoint()
            return True

        except Exception as e:
            log.error(f"Failed to get weights from {client_addr}: {e}")
            self.client_manager.remove_client(client)
            return False

    def initialize_client(self, client):
        """Initialize a new client with model data"""
        client_socket, client_addr = client

        try:
            log.debug(f"Sending weights to {client_addr}")
            CommunicationHandler.send_message(client_socket, self.weights)

            log.debug(f"Sending tokens to {client_addr}")
            batch_tokens, batch_info = next(self.tokens)
            CommunicationHandler.send_message(client_socket, batch_tokens)
            CommunicationHandler.send_message(client_socket, batch_info)

            log.debug(f"Sending optimizer state to {client_addr}")
            CommunicationHandler.send_message(client_socket, self.optimizer_values)

            log.debug(f"Sending config to {client_addr}")
            CommunicationHandler.send_message(client_socket, self.config)

            log.debug(f"Sending random seed to {client_addr}")
            client_seed = np.random.randint(0, self.seed_high)
            CommunicationHandler.send_message(client_socket, client_seed)

            log.info(f"Successfully sent initial data to {client_addr}")
            return True

        except Exception as e:
            log.error(f"Failed to initialize client {client_addr}: {e}")
            return False

    def process_new_clients(self):
        """Process and initialize new clients"""
        new_clients = self.client_manager.get_new_clients()
        if not new_clients:
            return

        # If we have active clients, get latest weights
        active_clients = self.client_manager.get_active_clients()
        if active_clients:
            # Try to get weights from any active client
            for client in active_clients:
                if self._get_weights_from_client(client):
                    break

        # Initialize each new client
        for client in new_clients:
            if self.initialize_client(client):
                self.client_manager.activate_client(client)
            else:
                self.client_manager.remove_client(client)

    def send_weight_request_status(self, client):
        """Send weight request status to a client"""
        client_socket, client_addr = client
        try:
            CommunicationHandler.send_message(client_socket, "dont need weights")
            return True
        except Exception as e:
            log.error(f"Failed to send status to {client_addr}: {e}")
            return False

    def send_tokens_to_client(self, client, send_new_tokens):
        """Send tokens to a client"""
        client_socket, client_addr = client
        try:
            if send_new_tokens:
                batch_tokens, batch_info = next(self.tokens)
                log.debug(f"Sending new tokens to {client_addr}")
                CommunicationHandler.send_message(client_socket, batch_tokens)
                CommunicationHandler.send_message(client_socket, batch_info)
            else:
                log.debug(f"Sending empty token indicator to {client_addr}")
                CommunicationHandler.send_message(
                    client_socket, np.array([], dtype=np.uint8)
                )
                CommunicationHandler.send_message(client_socket, [])
            return True
        except Exception as e:
            log.error(f"Failed to send tokens to {client_addr}: {e}")
            return False

    def receive_rewards_from_client(self, client):
        """Receive rewards from a client"""
        client_socket, client_addr = client
        client_rewards = []
        seed = None

        try:
            rewards_data = CommunicationHandler.receive_message(client_socket)
            if isinstance(rewards_data, np.ndarray) and rewards_data.size > 0:
                log.info(f"[{client_addr}] Received {rewards_data.size - 1:,} rewards.")
                seed = rewards_data[0].astype(np.uint32)
                client_rewards = rewards_data[1:]
                return True, client_rewards, seed
            else:
                log.warning(f"[{client_addr}] Received invalid rewards data")
                return False, [], None
        except Exception as e:
            log.error(f"Failed to receive rewards from {client_addr}: {e}")
            return False, [], None

    def send_normalized_rewards_to_client(
        self, client, normalized_rewards, reward_info, next_seed
    ):
        """Send normalized rewards to a client"""
        client_socket, client_addr = client
        try:
            # Send normalized rewards as bytes
            rewards_bytes = normalized_rewards.astype(np.float32).tobytes()
            CommunicationHandler.send_message(client_socket, rewards_bytes)

            # Send reward info and next seed
            response = {
                "reward_info": reward_info,
                "seed": next_seed,
            }
            CommunicationHandler.send_message(client_socket, response)
            log.info(f"Sent rewards and info to {client_addr}")
            return True
        except Exception as e:
            log.error(f"Failed to send normalized rewards to {client_addr}: {e}")
            return False

    def normalize_rewards(self, rewards):
        """Normalize rewards"""
        if not rewards:
            return np.array([], dtype=np.float32)

        rewards_array = np.array(rewards, dtype=np.float32)
        mean_reward = np.mean(rewards_array)
        std_dev = np.std(rewards_array)

        if np.isnan(mean_reward) or len(rewards) <= 1 or std_dev == 0:
            log.warning("Cannot normalize rewards. Using zeros.")
            return np.zeros(len(rewards), dtype=np.float32), mean_reward

        # Normalize rewards
        mul_val = 1.0 / (std_dev * float(len(rewards)) * self.config["sigma"])
        normalized = (rewards_array - mean_reward) * mul_val
        return normalized, mean_reward

    def run_training_step(self):
        """Run a single training step"""
        self.step_num += 1
        log.info(f"--- Starting Step {self.step_num} ---")

        # Process any new clients
        self.process_new_clients()

        # Check if checkpoint is needed
        if perf_counter() - self.last_checkpoint_time > self.config["checkPointTime"]:
            active_clients = self.client_manager.get_active_clients()
            if active_clients:
                # Try to get weights from first client for checkpoint
                self._get_weights_from_client(active_clients[0])

        # Send weight request status to all active clients
        active_clients = self.client_manager.get_active_clients()
        clients_to_remove = []

        for client in active_clients:
            if not self.send_weight_request_status(client):
                clients_to_remove.append(client)

        for client in clients_to_remove:
            self.client_manager.remove_client(client)

        # Determine if we should send new tokens
        send_new = self.config["newTokensInterval"] < 1e8 and (
            self.step_num % self.config["newTokensInterval"] == 0
        )

        # Send tokens to all active clients
        active_clients = self.client_manager.get_active_clients()
        clients_to_remove = []

        for client in active_clients:
            if not self.send_tokens_to_client(client, send_new):
                clients_to_remove.append(client)

        for client in clients_to_remove:
            self.client_manager.remove_client(client)

        # Collect rewards from all clients
        active_clients = self.client_manager.get_active_clients()
        all_rewards = []
        reward_info = []
        clients_to_remove = []

        for client in active_clients:
            success, client_rewards, seed = self.receive_rewards_from_client(client)
            if not success:
                clients_to_remove.append(client)
            elif len(client_rewards) > 0:
                all_rewards.extend(client_rewards)
                reward_info.append((len(client_rewards), seed))

        for client in clients_to_remove:
            self.client_manager.remove_client(client)

        # Process rewards
        if all_rewards:
            normalized_rewards, mean_reward = self.normalize_rewards(all_rewards)

            # Log mean reward
            log.info(f"Mean Reward: {mean_reward}")
            self._log_reward(mean_reward)

            # Generate next seeds for clients
            active_clients = self.client_manager.get_active_clients()
            seeds = np.random.randint(0, self.seed_high, len(active_clients))

            # Send normalized rewards to clients
            clients_to_remove = []
            for i, client in enumerate(active_clients):
                if not self.send_normalized_rewards_to_client(
                    client, normalized_rewards, reward_info, seeds[i]
                ):
                    clients_to_remove.append(client)

            for client in clients_to_remove:
                self.client_manager.remove_client(client)

        log.info(f"--- Finished Step {self.step_num} ---")

    def _log_reward(self, mean_reward):
        """Log reward to file"""
        loss_file_path = f"trainingRuns/{self.current_training_run}/loss.txt"
        try:
            with open(loss_file_path, "a", encoding="utf-8") as f:
                f.write(f"{mean_reward}\n")
        except IOError as e:
            log.error(f"Could not write to loss file {loss_file_path}: {e}")

    def run(self):
        """Main server loop"""
        self.running = True
        while self.running:
            if self.client_manager.count_clients() > 0:
                self.run_training_step()
            else:
                sleep(0.1)  # Wait if no clients

    def start(self, ip="0.0.0.0", port=55551):
        """Start the server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server_socket.bind((ip, port))
            server_socket.settimeout(2.0)
        except OSError as e:
            log.critical(f"Failed to bind server to {ip}:{port}: {e}")
            return False

        server_socket.listen(5)
        try:
            host_ip = socket.gethostbyname(socket.gethostname())
            log.info(f"[LISTENING] Server is listening on {host_ip}:{port}")
        except socket.gaierror:
            log.info(f"[LISTENING] Server is listening on {ip}:{port}")

        # Start handler thread
        handler_thread = threading.Thread(target=self.run, daemon=True)
        handler_thread.start()

        try:
            while self.running:
                try:
                    client_socket, addr = server_socket.accept()
                    client_socket.settimeout(self.config["timePerStep"] + 15)
                    log.info(f"[CONNECTION] Accepted connection from {addr}")
                    self.client_manager.add_new_client(client_socket, addr)
                    log.info(
                        f"[ACTIVE CONNECTIONS] {self.client_manager.count_clients()}"
                    )
                except socket.timeout:
                    continue
                except OSError as e:
                    log.error(f"Error accepting connection: {e}")
                    sleep(1)
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received. Shutting down server...")
        finally:
            self.running = False
            log.info("Waiting for handler thread to finish...")
            handler_thread.join(5.0)

            # Clean up all clients
            for client in self.client_manager.get_active_clients():
                self.client_manager.remove_client(client)

            # Close server socket
            server_socket.close()
            log.info("Server socket closed.")
            return True


def start_server():
    """Entry point function"""
    server = TrainingServer()
    server.start()


if __name__ == "__main__":
    start_server()
