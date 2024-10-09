import socket
from basicCommunicationUtils import *
from utilitiesModel import *
from time import perf_counter


def start_client():
    print("Started client")
    # Server settings
    server_ip = "130.215.211.30"
    server_port = 55551

    # Create a socket object
    print("Connecting to server")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    connected = True
    print("Connected")

    # Receive initial data
    # receive weights
    print("Waiting to receive weights")
    weights, valid = receiveData(client, "np.float32", "SERVER")
    weights = weights.copy()
    grad = np.zeros_like(weights)
    # receive tokens
    print("Waiting to receive tokens")
    tokens, valid = receiveData(client, "np.uint16", "SERVER")
    # receive tokens info
    print("Waiting to receive token info")
    tokenInfo, valid = receiveData(client, "pickle", "SERVER")
    batchTokens = []
    for length in tokenInfo:
        batchTokens.append(tokens[:length])
        tokens = tokens[length:]
    totalNumTokens = sum(tokenInfo)
    # receive optimizer state
    print("Waiting to receive optimizer values")
    optimizerValues, valid = receiveData(client, "np.float32", "SERVER")
    # receive config
    print("Waiting to receive config")
    config, valid = receiveData(client, "pickle", "SERVER")
    # receive random seed
    print("Waiting to receive random seed")
    seed, valid = receiveData(client, "pickle", "SERVER")
    print(f"Received initial data")

    print("Initializing optimizer")
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

    print("Initializing model")
    if config["modelType"] == "critic":
        model = ChatCritic()
        tokens = [
            [tokens, 1],
        ]
    else:
        model = ChatModel()

    start = perf_counter()

    while connected:
        print(weights)
        # Receive weight request
        print(f"Checking for weights request")
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
            print(f"Sent weights to server")

            request, valid = receiveData(
                client, "text", "SERVER"
            )  # just receive the "dont need weights" that is sent to everyone

        # Run trials
        print(f"Running trials for {config['timePerStep']}s on {totalNumTokens} tokens")
        rewards = [seed]
        numTrials = 0
        np.random.seed(seed)
        while perf_counter() - start < config["timePerStep"]:
            loss = model.getLoss(
                weights + np.random.randn(weights.shape[0]) * config["sigma"],
                batchTokens,
                config["hiddenSize"],
                config["vocabSize"],
            )
            rewards.append(loss)
            numTrials += 1
        rewards = np.array(rewards).astype(np.float32).tobytes()
        sendBytes(client, rewards, "SERVER")
        print(f"Sent {numTrials:,} rewards to server")

        # Receive normalize rewards
        print(f"Waiting for normalized rewards")
        normalizedRewards, valid = receiveData(client, "np.float32", "SERVER")
        data, valid = receiveData(client, "pickle", "SERVER")
        print(f"Received normalized rewards")
        start = perf_counter()

        reward_info = data["reward_info"]
        seed = data["seed"]

        # Update weights
        print(f"Updating weights")
        rewardNum = 0
        grad.fill(0)
        for nTrials, trialSeed in reward_info:
            np.random.seed(trialSeed)
            mulVal = config["sigma"] / float(
                nTrials
            )  # normalize the grad by the number of samples per example
            for trial in range(nTrials):
                grad += (
                    np.random.randn(weights.shape[0])
                    * mulVal
                    * normalizedRewards[rewardNum]
                )
                rewardNum += 1
        grad *= 1.0 / len(reward_info)
        grad = optimizer.getGrad(grad)
        weights -= grad

    client.close()


if __name__ == "__main__":
    start_client()
