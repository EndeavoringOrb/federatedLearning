from twisted.internet import reactor, protocol
from twisted.protocols import basic
import numpy as np
import pickle
from utilitiesMisc import *
from utilitiesModel import *
from utilitiesData import *
from squad.squad import dataLoader

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
        if (
            len(self.factory.reward_info) >= len(self.factory.clients)
            and len(self.factory.reward_info) > 0
        ):
            self.factory.process_rewards()

    def lineReceived(self, line):
        try:
            data = np.frombuffer(line, dtype=np.float32)
            if data[0] < 0:
                self.factory.config["stepNum"] = int(data[1])
                self.factory.stepNum = self.factory.config["stepNum"]
                if data.shape[0] != 2 + 3 * self.factory.nParams:
                    print(
                        f"ERROR: Wrong number of elements for optimizer state and weights received"
                    )
                    self.transport.loseConnection()
                    return
                nParams = self.factory.nParams
                self.factory.optimizerWeights = data[2 : 2 + 2 * nParams]
                self.factory.weights = data[2 + 2 * nParams :]
                self.factory.newWeights = True
                print(f"Recieved optimizer state and model weights {currentTime()}")
                np.save(
                    "weights/model.npy",
                    np.concatenate(
                        (
                            [
                                self.factory.config["hiddenSize"],
                                self.factory.config["vocabSize"],
                            ],
                            self.factory.weights,
                        )
                    ),
                )
            else:
                self.handle_rewards(data)
        except Exception as e:
            print(f"ERROR: {e}")
            print(line)
            self.transport.loseConnection()

    def send_initial_data(self):
        data = self.factory.weights.tobytes()
        self.sendLine(data)
        self.sendLine(pickle.dumps(self.factory.getTokens()))
        data = self.factory.optimizerWeights.tobytes()
        self.sendLine(data)
        self.sendLine(
            pickle.dumps((self.factory.getConfig(), np.random.randint(0, seedHigh)))
        )
        print(f"Sent initial data to {self}")

    def handle_rewards(self, rewards):
        seed = rewards[0].astype(np.uint32)
        self.factory.all_rewards.extend(rewards[1:])
        self.factory.reward_info.append((len(rewards), seed))

        print(
            f"Recieved rewards [{len(self.factory.reward_info)}/{len(self.factory.clients)}] {currentTime()}"
        )

        if len(self.factory.reward_info) >= len(self.factory.clients):
            self.factory.process_rewards()


class ServerFactory(protocol.Factory):
    def __init__(self):
        self.clients = []
        self.newClients = []
        self.config = {
            "timePerStep": 10,
            "learningRate": 0.01,
            "sigma": 0.1,
            "hiddenSize": 8,
            "vocabSize": 128,
            "beta1": 0.9,
            "beta2": 0.999,
            "stepNum": 0,
            "modelType": "chat",
        }
        if self.config["modelType"] == "critic":
            self.model = ChatCritic()
        else:
            self.model = ChatModel()
        self.weights = self.model.getWeights(
            self.config["hiddenSize"], self.config["vocabSize"]
        )
        self.nParams = self.weights.shape[0]
        self.optimizerWeights = np.zeros(self.nParams * 2).astype(np.float32)
        self.newWeights = False
        self.all_rewards = []
        self.reward_info = []
        self.stepNum = 0

        self.tokenLoader = dataLoader()

    def getConfig(self):
        return self.config

    def getTokens(self):
        return next(self.tokenLoader)

    def buildProtocol(self, addr):
        return ServerProtocol(self)

    def process_rewards(self):
        numRewards = len(self.all_rewards)
        print(f"Processing rewards ({numRewards}) {currentTime()}")
        if numRewards == 1:
            normalizedRewards = np.zeros(1)
            print(f"Mean Reward: {0}")
        else:
            mean = np.mean(self.all_rewards)
            print(f"Mean Reward: {mean}")
            normalizedRewards = (np.array(self.all_rewards) - mean) / np.std(
                self.all_rewards
            )

        if self.newWeights:
            print(f"Sending weights to new clients {currentTime()}")
            for client in self.newClients:
                client.send_initial_data()
            self.newWeights = False

            self.clients.extend(self.newClients)
            self.newClients = []

        print(f"Sending rewards and info for next iteration {currentTime()}")
        seeds = np.random.randint(0, seedHigh, len(self.clients))
        normalizedRewardsBytes = normalizedRewards.astype(np.float32).tobytes()
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

        print(f"Finished step {self.stepNum:,} {currentTime()}\n\n")
        self.all_rewards = []
        self.reward_info = []
        self.stepNum += 1


if __name__ == "__main__":
    port = 54329
    reactor.listenTCP(port, ServerFactory())
    print(f"Server started on port {port}")
    reactor.run()
