from twisted.internet import reactor, protocol
from twisted.protocols import basic
from time import sleep, perf_counter
import numpy as np
import pickle
from helperFuncs import *


class ClientProtocol(basic.LineReceiver):
    def connectionMade(self):
        print(f"Connected to server {currentTime()}")
        self.factory.client = self
        self.factory.haveWeights = False
        self.factory.haveConfig = False
        self.factory.haveTokens = False
        self.factory.haveNormalizedRewards = False

    def lineReceived(self, line):
        if not self.factory.haveWeights:
            self.factory.weights = np.frombuffer(line, dtype=np.float32).copy()
            self.factory.haveWeights = True
            print(f"Received initial weights {currentTime()}")
        elif not self.factory.haveTokens:
            self.factory.tokens = np.frombuffer(line, dtype=np.uint8)
            self.factory.haveTokens = True
            print(f"Received initial tokens {currentTime()}")
        elif not self.factory.haveConfig:
            config, seed = pickle.loads(line)
            self.factory.config = config
            self.factory.haveConfig = True
            print(f"Received initial config {currentTime()}")
            self.run_trials(seed)
        elif not self.factory.haveNormalizedRewards:
            self.factory.normalizedRewards = np.frombuffer(line, dtype=np.float32)
            self.factory.haveNormalizedRewards = True
            print(f"Received normalized rewards {currentTime()}")
        else:
            data = pickle.loads(line)
            self.handle_normalized_rewards(data)

    def run_trials(self, seed):
        print(f"Running trials {currentTime()}")
        start = perf_counter()
        # Simulate running trials
        rewards = [seed]
        while perf_counter() - start < self.factory.config['timePerStep']:
            rewards.append(np.random.randn())
            sleep(0.01)
        rewards = np.array(rewards).astype(np.float32)
        print(f"Finished running trials {currentTime()}")

        self.sendLine(rewards.tobytes())
        print(f"Sent rewards to server {currentTime()}")

    def handle_normalized_rewards(self, data):
        self.factory.haveNormalizedRewards = False  # Reset flag
        reward_info = data["reward_info"]
        needWeights = data["needWeights"]
        seed = data["seed"]

        print(f"Mean reward: {np.mean(self.factory.normalizedRewards)}")

        # Update weights (simplified example)
        self.factory.weights += np.random.randn(*self.factory.weights.shape) * 0.01

        # Send updated weights back to server
        if needWeights:
            response = {"weights": self.factory.weights.tolist()}
            self.sendLine(pickle.dumps(response))
            print(f"Sent weights to server {currentTime()}")

        self.run_trials(seed)


class ClientFactory(protocol.ClientFactory):
    protocol = ClientProtocol

    def clientConnectionFailed(self, connector, reason):
        print("Connection failed")
        print(f"Connector: {connector}")
        print(f"Reason: {reason}")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("Connection lost")
        print(f"Connector: {connector}")
        print(f"Reason: {reason}")
        reactor.stop()


if __name__ == "__main__":
    factory = ClientFactory()
    reactor.connectTCP("localhost", 8000, factory)
    reactor.run()
