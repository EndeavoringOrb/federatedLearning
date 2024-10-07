import socket
from basicCommunicationUtils import *
from utilitiesModel import *
from time import perf_counter
import os


def start_client():
    # Server settings
    server_ip = "130.215.211.30"
    server_port = 55551

    # Create a socket object
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    connected = True

    # Receive initial data
    # receive weights
    weights, valid = receiveData(client, "np.float32", "SERVER")
    weights = weights.copy()
    grad = np.zeros_like(weights)
    # receive tokens
    tokens, valid = receiveData(client, "np.uint8", "SERVER")
    # receive optimizer state
    optimizerValues, valid = receiveData(client, "np.float32", "SERVER")
    # receive config
    config, valid = receiveData(client, "pickle", "SERVER")
    # receive random seed
    seed, valid = receiveData(client, "pickle", "SERVER")

    optimizer = AdamOptimizer(
        weights.shape[0],
        config["learningRate"],
        config["beta1"],
        config["beta2"],
    )
    optimizer.m = optimizerValues[: weights.shape[0]]
    optimizer.v = optimizerValues[weights.shape[0] :]
    optimizer.t = config["stepNum"]
    for i in range(optimizer.t):
        optimizer.beta1Power *= optimizer.beta1
        optimizer.beta2Power *= optimizer.beta2

    if config["modelType"] == "critic":
        model = ChatCritic()
        tokens = [
            [tokens, 1],
        ]
    else:
        model = ChatModel()

    while connected:
        print(weights)
        # Receive weight request
        request, valid = receiveData(client, "text", "SERVER")
        if request == "need weights":
            data = (
                np.concatenate(
                    [
                        [optimizer.t],
                        optimizer.m,
                        optimizer.v,
                        weights,
                    ]
                )
                .astype(np.float32)
                .tobytes()
            )
            sendBytes(client, data, "SERVER")

            request, valid = receiveData(
                client, "text", "SERVER"
            )  # just receive the "dont need weights" that is sent to everyone

        # Run trials
        print(f"Running trials for {config['timePerStep']}s")
        start = perf_counter()
        rewards = [seed]
        np.random.seed(seed)
        while perf_counter() - start < config["timePerStep"]:
            loss = model.getLoss(
                weights + np.random.randn(weights.shape[0]) * config["sigma"],
                [tokens],
                config["hiddenSize"],
                config["vocabSize"],
            )
            rewards.append(loss)
        rewards = np.array(rewards).astype(np.float32).tobytes()
        sendBytes(client, rewards, "SERVER")

        # Receive normalize rewards
        normalizedRewards, valid = receiveData(client, "np.float32", "SERVER")
        data, valid = receiveData(client, "pickle", "SERVER")

        reward_info = data["reward_info"]
        seed = data["seed"]

        # Update weights
        rewardNum = 0
        grad.fill(0)
        for nTrials, trialSeed in reward_info:
            np.random.seed(trialSeed)
            for trial in range(nTrials - 1):
                grad += (
                    np.random.randn(weights.shape[0])
                    * config["sigma"]
                    * normalizedRewards[rewardNum]
                )
                rewardNum += 1
        grad *= 1.0 / rewardNum
        grad = optimizer.getGrad(grad)
        weights -= grad

        #fileNum = len(os.listdir("weights"))
        #np.save(f"weights/{fileNum}.npy", weights)

    client.close()


if __name__ == "__main__":
    start_client()
