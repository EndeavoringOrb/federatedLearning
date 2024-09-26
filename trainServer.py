from twisted.internet import reactor, protocol
from twisted.protocols import basic
import numpy as np
import pickle
from helperFuncs import *

seedHigh = 1_000_000


class ServerProtocol(basic.LineReceiver):
    def __init__(self, factory):
        self.factory = factory

    def connectionMade(self):
        if len(self.factory.clients) == 0:
            self.factory.clients.append(self)
            print(f"New client connected {currentTime()}")
            print(
                f"Total clients: {len(self.factory.clients) + len(self.factory.newClients)} ({len(self.factory.newClients)} new)"
            )
            self.send_initial_data()
        else:
            self.factory.newClients.append(self)
            print(f"New client connected {currentTime()}")
            print(
                f"Total clients: {len(self.factory.clients) + len(self.factory.newClients)} ({len(self.factory.newClients)} new)"
            )

    def connectionLost(self, reason):
        try:
            self.factory.clients.remove(self)
        except ValueError:
            pass
        try:
            self.factory.newClients.remove(self)
        except ValueError:
            pass
        print(f"Client disconnected {currentTime()}")
        print(f"Reason: {reason}")
        print(
            f"Total clients: {len(self.factory.clients) + len(self.factory.newClients)} ({len(self.factory.newClients)} new)"
        )

    def lineReceived(self, line):
        try:
            data = pickle.loads(line)
            self.factory.weights = np.array(data["weights"])
            self.factory.newWeights = True
            print(f"Recieved weights {currentTime()}")
        except Exception:
            self.handle_rewards(line)

    def send_initial_data(self):
        data = self.factory.weights.tobytes()
        self.sendLine(data)
        self.sendLine(self.factory.getTokens().tobytes())
        self.sendLine(
            pickle.dumps((self.factory.getConfig(), np.random.randint(0, seedHigh)))
        )
        print(f"Sent initial data to {self}")

    def handle_rewards(self, data):
        try:
            rewards = np.frombuffer(data, dtype=np.float32)
            seed = rewards[0].astype(np.uint32)
            self.factory.all_rewards.extend(rewards[1:])
            self.factory.reward_info.append((len(rewards), seed))

            print(
                f"Recieved rewards [{len(self.factory.reward_info)}/{len(self.factory.clients)}] {currentTime()}"
            )

            if len(self.factory.reward_info) == len(self.factory.clients):
                self.factory.process_rewards()
        except Exception as e:
            print(f"ERROR: {e}")
            print(data)
            self.transport.loseConnection()


class ServerFactory(protocol.Factory):
    def __init__(self):
        self.clients = []
        self.newClients = []
        self.weights = np.random.randn(100)  # Example initial weights
        self.newWeights = False
        self.config = {
            "timePerStep": 2.5,
            "sigma": 0.1,
        }
        self.all_rewards = []
        self.reward_info = []
        self.stepNum = 0

    def getConfig(self):
        return self.config

    def getTokens(self):
        return np.random.randint(0, 256, 100, dtype=np.uint8)

    def buildProtocol(self, addr):
        return ServerProtocol(self)

    def process_rewards(self):
        print(f"Processing rewards {currentTime()}")
        normalizedRewards = (
            np.array(self.all_rewards) - np.mean(self.all_rewards)
        ) / np.std(self.all_rewards)

        if self.newWeights:
            print(f"Sending weights to new clients {currentTime()}")
            for client in self.newClients:
                client.send_initial_data()
            self.newWeights = False

            self.clients.extend(self.newClients)
            self.newClients = []

        print(f"Sending rewards and info for next iteration {currentTime()}")
        seeds = np.random.randint(0, seedHigh, len(self.clients))
        normalizedRewardsBytes = normalizedRewards.tobytes()
        for i, client in enumerate(self.clients):
            response = {
                "reward_info": self.reward_info,
                "needWeights": (i == 0) and (len(self.newClients) > 0),
                "seed": seeds[i],
            }
            if (i == 0) and (len(self.newClients) > 0):
                print(f"Requesting weights from {client} {currentTime()}")
            client.sendLine(normalizedRewardsBytes)
            client.sendLine(pickle.dumps(response))
            print(f"Sent rewards and info [{i+1}/{len(self.clients)}] {currentTime()}")

        print(f"Finished step {self.stepNum:,} {currentTime()}")
        self.all_rewards = []
        self.reward_info = []
        self.stepNum += 1


if __name__ == "__main__":
    reactor.listenTCP(8000, ServerFactory())
    print("Server started on port 8000")
    reactor.run()
