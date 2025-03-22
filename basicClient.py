import socket
from utilities.communication import *
from utilities.model import *
from time import perf_counter


def start_client(server_ip, server_port):
    print("Started client")

    # Init trackers
    stepStart = perf_counter()
    timePerTrial = 0  # we measure the average time per trial and then use this to estimate whether running the current trial will take us over the allotted time

    # Create a socket object
    print("Connecting to server")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    connected = True
    print("Connected")

    # Receive initial data
    ## receive weights
    print("Waiting to receive weights")
    weights, valid = receiveData(client, "np.float32", "SERVER")
    if not valid:
        exit(0)
    weights = weights.copy()
    grad: np.ndarray = np.zeros_like(weights)
    ## receive tokens
    print("Waiting to receive tokens")
    tokens, valid = receiveData(client, "np.uint8", "SERVER")
    if not valid:
        exit(0)
    ## receive tokens info
    print("Waiting to receive token info")
    tokenInfo, valid = receiveData(client, "pickle", "SERVER")
    if not valid:
        exit(0)
    batchTokens = []
    for length in tokenInfo:
        batchTokens.append(tokens[:length])
        tokens = tokens[length:]
    totalNumTokens = sum(tokenInfo)
    maxNumTokens = float(max(tokenInfo))
    ## receive optimizer state
    print("Waiting to receive optimizer values")
    optimizerValues, valid = receiveData(client, "np.float32", "SERVER")
    if not valid:
        exit(0)
    ## receive config
    print("Waiting to receive config")
    config, valid = receiveData(client, "pickle", "SERVER")
    if not valid:
        exit(0)
    ## receive random seed
    print("Waiting to receive random seed")
    seed, valid = receiveData(client, "pickle", "SERVER")
    if not valid:
        exit(0)
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
    if config["modelType"] == "chat":
        model = ChatModel()
    else:
        model = ChatModel()

    firstStep = True

    while connected:
        print()
        print(weights)
        # Receive weight request
        print(f"Checking for weights request ", end="")
        start = perf_counter()
        request, valid = receiveData(client, "text", "SERVER")
        if not valid:
            exit(0)
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
            if not valid:
                exit(0)
        elapsed = perf_counter() - start
        print(f"{elapsed}s")

        # Receieve new tokens
        # receive tokens
        print("Waiting to receive tokens ", end="")
        start = perf_counter()
        tokens, valid = receiveData(client, "np.uint8", "SERVER")
        if not valid:
            exit(0)
        elapsed = perf_counter() - start
        print(f"{elapsed}s")
        # receive tokens info
        print("Waiting to receive token info ", end="")
        start = perf_counter()
        tokenInfo, valid = receiveData(client, "pickle", "SERVER")
        if not valid:
            exit(0)
        elapsed = perf_counter() - start
        print(f"{elapsed}s")

        trialStart = perf_counter()

        # update batchTokens if we were actually sent new tokens
        if tokenInfo != []:
            batchTokens = []
            for length in tokenInfo:
                batchTokens.append(tokens[:length])
                tokens = tokens[length:]
            totalNumTokens = sum(tokenInfo)
            maxNumTokens = float(max(tokenInfo))

        # Run trials
        print(
            f"Running trials for {config['timePerStep']}s on {totalNumTokens} tokens. Avg. batch size: {totalNumTokens / maxNumTokens}"
        )
        rewards = [seed]
        numTrials = 0
        np.random.seed(seed)

        # while (estimated end of next step < time limit)
        while (
            perf_counter()
            - stepStart
            + (
                (perf_counter() - trialStart) / numTrials
                if numTrials > 0
                else timePerTrial
            )
            < config["timePerStep"]
            and not firstStep
        ):
            loss = model.getLossBatched(
                weights + np.random.randn(weights.shape[0]) * config["sigma"],
                batchTokens,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )
            rewards.append(loss)
            numTrials += 1
        trialEnd = perf_counter()
        elapsed = trialEnd - trialStart
        timePerTrial = elapsed / numTrials if numTrials > 0 else 0
        print(f"{trialEnd-trialStart}s")
        print(f"{(len(rewards)*totalNumTokens)/(trialEnd - trialStart)} tok/sec")

        print(f"Sending {numTrials:,} rewards to server ", end="")
        start = perf_counter()
        rewards = np.array(rewards).astype(np.float32).tobytes()
        sendBytes(client, rewards, "SERVER")
        elapsed = perf_counter() - start
        print(f"{elapsed}s")

        # Receive normalize rewards
        print(f"Waiting for normalized rewards ", end="")
        start = perf_counter()
        normalizedRewards, valid = receiveData(client, "np.float32", "SERVER")
        if not valid:
            exit(0)
        data, valid = receiveData(client, "pickle", "SERVER")
        if not valid:
            exit(0)
        reward_info = data["reward_info"]
        seed = data["seed"]
        elapsed = perf_counter() - start
        print(f"{elapsed}s")
        stepStart = perf_counter()

        # Update weights
        print(f"Updating weights ", end="")
        start = perf_counter()
        rewardNum = 0
        grad.fill(0)
        totalNTrials = float(sum([item[0] for item in reward_info]))
        for nTrials, trialSeed in reward_info:
            np.random.seed(trialSeed)
            for trial in range(nTrials):
                grad += np.random.randn(weights.shape[0]) * normalizedRewards[rewardNum]
                rewardNum += 1
        if config["optimizer"] == "adam":
            grad = optimizer.getGrad(grad)
        else:
            grad *= config["learningRate"]
        gradNorm = np.sqrt((grad**2).sum())
        weights -= grad
        elapsed = perf_counter() - start
        print(f"{elapsed}s")
        print(f"Grad Norm: {gradNorm}")

        firstStep = False

    client.close()


if __name__ == "__main__":
    # Server settings
    server_ip = "130.215.13.29"
    server_port = 55551
    start_client(server_ip, server_port)
