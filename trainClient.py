from twisted.internet import reactor, protocol
from twisted.protocols import basic
from time import sleep, perf_counter
import numpy as np
import pickle

from utilitiesMisc import *
from utilitiesModel import *
from utilitiesData import *
from squad.squad import *


class ClientProtocol(basic.LineReceiver):
    def connectionMade(self):
        log("Connected to server")
        self.factory.client = self
        self.factory.haveWeights = False
        self.factory.haveOptimizerWeights = False
        self.factory.haveConfig = False
        self.factory.haveTokens = False
        self.factory.haveNormalizedRewards = False

    def lineReceived(self, line):
        if not self.factory.haveWeights:
            self.factory.weights = np.frombuffer(line, dtype=np.float32).copy()
            self.factory.grad = np.zeros_like(self.factory.weights)
            self.factory.haveWeights = True
            log(f"Received initial weights (shape {self.factory.weights.shape})")
        elif not self.factory.haveTokens:
            self.factory.tokens = formatDataExample(pickle.loads(line), self.factory.searcher)
            self.factory.haveTokens = True
            log(f"Received initial tokens (shape {self.factory.tokens.shape})")
        elif not self.factory.haveOptimizerWeights:
            self.factory.optimizerWeights = np.frombuffer(line, dtype=np.float32).copy()
            self.factory.haveOptimizerWeights = True
            log(
                f"Received initial optimizer weights (shape {self.factory.optimizerWeights.shape})"
            )
        elif not self.factory.haveConfig:
            config, seed = pickle.loads(line)
            self.factory.config = config
            self.factory.haveConfig = True
            self.factory.optimizer = AdamOptimizer(
                self.factory.weights.shape[0],
                self.factory.config["learningRate"],
                self.factory.config["beta1"],
                self.factory.config["beta2"],
            )
            self.factory.optimizer.m = self.factory.optimizerWeights[
                : self.factory.weights.shape[0]
            ]
            self.factory.optimizer.v = self.factory.optimizerWeights[
                self.factory.weights.shape[0] :
            ]
            self.factory.optimizer.t = self.factory.config["stepNum"]
            for i in range(self.factory.optimizer.t):
                self.factory.optimizer.beta1Power *= self.factory.optimizer.beta1
                self.factory.optimizer.beta2Power *= self.factory.optimizer.beta2

            if self.factory.config["modelType"] == "critic":
                self.factory.model = ChatCritic()
                self.factory.tokens = [
                    [self.factory.tokens, 1],
                ]
            else:
                self.factory.model = ChatModel()
            log("Received initial config")
            self.run_trials(seed)
        elif not self.factory.haveNormalizedRewards:
            self.factory.normalizedRewards = np.frombuffer(line, dtype=np.float32)
            self.factory.haveNormalizedRewards = True
            log("Received normalized rewards")
        else:
            data = pickle.loads(line)
            self.handle_normalized_rewards(data)

    def run_trials(self, seed):
        log("Running trials")
        start = perf_counter()
        # Simulate running trials
        rewards = [seed]
        np.random.seed(seed)
        while perf_counter() - start < self.factory.config["timePerStep"]:
            loss = self.factory.model.getLoss(
                self.factory.weights
                + np.random.randn(self.factory.weights.shape[0])
                * self.factory.config["sigma"],
                self.factory.tokens,
                self.factory.config["hiddenSize"],
                self.factory.config["vocabSize"],
            )
            rewards.append(loss)
        rewards = np.array(rewards).astype(np.float32)
        log("Finished running trials")

        self.sendLine(rewards.tobytes())
        log("Sent rewards to server")

    def handle_normalized_rewards(self, data):
        self.factory.haveNormalizedRewards = False  # Reset flag
        reward_info = data["reward_info"]
        needWeights = data["needWeights"]
        seed = data["seed"]

        # Update weights
        rewardNum = 0
        self.factory.grad.fill(0)
        for nTrials, trialSeed in reward_info:
            np.random.seed(trialSeed)
            for trial in range(nTrials - 1):
                self.factory.grad += (
                    np.random.randn(self.factory.weights.shape[0])
                    * self.factory.config["sigma"]
                    * self.factory.normalizedRewards[rewardNum]
                )
                rewardNum += 1
        self.factory.grad = self.factory.optimizer.getGrad(self.factory.grad)
        self.factory.weights -= self.factory.grad

        # fileNum = len(os.listdir("weights"))
        # np.save(f"weights/{fileNum}.npy", self.factory.weights)

        # Send updated weights back to server
        if needWeights:
            self.sendLine(
                np.concatenate(
                    [
                        [-1.0, self.factory.optimizer.t],
                        self.factory.optimizer.m,
                        self.factory.optimizer.v,
                        self.factory.weights,
                    ]
                )
                .astype(np.float32)
                .tobytes()
            )
            log("Sent weights to server")

        self.run_trials(seed)


class ClientFactory(protocol.ClientFactory):
    protocol = ClientProtocol

    def __init__(self) -> None:
        super().__init__()
        self.searcher = ArticleSearcher("C:/Users/aaron/CODING/wikiTalk/tokenData")

    def clientConnectionFailed(self, connector, reason):
        log("Connection failed")
        log(f"Connector: {connector}")
        log(f"Reason: {reason}")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        log("Connection lost")
        log(f"Connector: {connector}")
        log(f"Reason: {reason}")
        reactor.stop()


if __name__ == "__main__":
    port = 54329
    factory = ClientFactory()
    reactor.connectTCP("130.215.211.30", port, factory)
    reactor.run()
