import socket
from time import perf_counter
import numpy as np
import socket
import struct
import pickle
from datetime import datetime

headerFormat = "I"
headerSize = 4
DEBUG = False
BUFFER_SIZE = 1024


def currentTime():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(message):
    if DEBUG:
        print(f"{currentTime()}|{message}", flush=True)


def recvall(sock: socket.socket, size):
    received_chunks = []
    remaining = size
    while remaining > 0:
        received = sock.recv(min(remaining, BUFFER_SIZE))
        if not received:
            raise Exception("unexpected EOF")
        received_chunks.append(received)
        remaining -= len(received)
    return b"".join(received_chunks)


def sendBytes(connection: socket.socket, data: bytes, addr):
    log(f"SENDING DATA")
    dataLength = len(data)

    # send header
    connection.send(struct.pack(headerFormat, dataLength))
    log(f"  Sent header specifying length of {len(data)} bytes")

    # chunk data
    log(f"  Chunking bytes")
    dataChunks = []
    while data:
        dataChunks.append(data[:BUFFER_SIZE])
        data = data[BUFFER_SIZE:]
    numChunks = len(dataChunks)

    # send chunks
    log(f"  Sending chunks")
    for i, chunk in enumerate(dataChunks):
        connection.send(chunk)
        log(f"  Sent chunk {i+1}/{numChunks} with length {len(chunk)}")


def receiveData(connection: socket.socket, dataType: str, addr):
    log("RECEIVING DATA")
    msg = recvall(connection, headerSize)
    msgLength = struct.unpack(headerFormat, msg)[0]
    if not msg:
        log(f"  Received empty message from {addr}")
        return msg, False
    log(f"  Receiving message with length {msgLength}")

    # Receive message bytes in chunks
    msg = recvall(connection, msgLength)
    log(f"  Received {len(msg)}/{msgLength} bytes")

    # decode based on datatype
    if dataType == "text":
        data = msg.decode("utf-8")
    elif dataType == "np.uint8":
        data = np.frombuffer(msg, np.uint8)
    elif dataType == "np.uint16":
        data = np.frombuffer(msg, np.uint16)
    elif dataType == "np.float32":
        data = np.frombuffer(msg, np.float32)
    elif dataType == "pickle":
        data = pickle.loads(msg)
    else:
        data = msg.decode("utf-8")  # just try normal decode

    # return decoded data
    return data, True


def softmax(x):
    x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    invSum = 1.0 / np.sum(x)
    return x * invSum


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    np.maximum(x, 0, x)


class ChatModel:
    def __init__(self) -> None:
        pass

    # @profile
    def getPred(self, weights, state, hiddenSize, vocabSize, nLayers):
        out = state @ weights[
            hiddenSize
            + nLayers * (hiddenSize * hiddenSize + hiddenSize * vocabSize) : hiddenSize
            + nLayers * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
            + hiddenSize * (hiddenSize * 4)
        ].reshape(hiddenSize, hiddenSize * 4)
        relu(out)
        out = out @ weights[
            hiddenSize
            + nLayers * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
            + hiddenSize * (hiddenSize * 4) :
        ].reshape(hiddenSize * 4, vocabSize)
        return out

    def getNextState(self, weights, state, token, hiddenSize, vocabSize, nLayers):
        for i in range(nLayers):
            np.tanh(
                state
                + (
                    state
                    @ weights[
                        hiddenSize
                        + i
                        * (
                            hiddenSize * hiddenSize + hiddenSize * vocabSize
                        ) : hiddenSize
                        + i * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
                        + hiddenSize * hiddenSize
                    ].reshape(hiddenSize, hiddenSize)
                )
                * weights[
                    hiddenSize
                    + hiddenSize * hiddenSize
                    + i * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
                    + hiddenSize * token : hiddenSize
                    + hiddenSize * hiddenSize
                    + i * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
                    + hiddenSize * (token + 1)
                ],
                state,
            )
        return state

    # @profile
    def getNextStateBatched(
        self, weights, state, tokens, hiddenSize, vocabSize, nLayers
    ):
        for i in range(nLayers):
            np.tanh(
                state
                + (
                    state
                    @ weights[
                        hiddenSize
                        + i
                        * (
                            hiddenSize * hiddenSize + hiddenSize * vocabSize
                        ) : hiddenSize
                        + i * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
                        + hiddenSize * hiddenSize
                    ].reshape(hiddenSize, hiddenSize)
                )
                * weights[
                    hiddenSize
                    + hiddenSize * hiddenSize
                    + i
                    * (hiddenSize * hiddenSize + hiddenSize * vocabSize) : hiddenSize
                    + hiddenSize * hiddenSize
                    + i * (hiddenSize * hiddenSize + hiddenSize * vocabSize)
                    + hiddenSize * vocabSize
                ].reshape(vocabSize, hiddenSize)[tokens],
                state,
            )
        return state

    def getLoss(self, weights, tokens, hiddenSize, vocabSize, nLayers):
        loss = 0.0
        numTokens = 0

        for chunk in tokens:
            state = weights[:hiddenSize]
            numTokens += len(chunk)

            for token in chunk:
                token = token.astype(np.uint32)
                preds = self.getPred(weights, state, hiddenSize, vocabSize, nLayers)
                preds = np.exp(preds - np.max(preds))
                loss -= np.log(preds[token] / np.sum(preds))
                state = self.getNextState(
                    weights, state, token, hiddenSize, vocabSize, nLayers
                )

        return loss / numTokens

    # @profile
    def getLossBatched(
        self, weights, tokens: list[list], hiddenSize, vocabSize, nLayers
    ):
        loss = 0.0
        batchSize = len(tokens)

        state = weights[:hiddenSize].reshape(1, -1).repeat(batchSize, 0)
        chunkLengths = np.asarray([len(chunk) for chunk in tokens])
        maxLength = max(chunkLengths)
        numTokens = sum(chunkLengths)
        indices = np.arange(batchSize)

        for i in range(maxLength):
            # Remove sequences that are finished
            # TODO: optimize this by doing a nested for loop with the chunkLengths
            removeIndices: np.ndarray = np.where(chunkLengths == i)[0]
            if len(removeIndices) > 0:
                removeIndices[::-1].sort()
                for idx in removeIndices:
                    tokens.pop(idx)
                chunkLengths = np.delete(chunkLengths, removeIndices)
                state = np.delete(state, removeIndices, axis=0)
                indices = np.arange(len(chunkLengths))

            # Get tokens for current step
            currentTokens = [chunk[i] for chunk in tokens]

            # Get preds
            preds = self.getPred(weights, state, hiddenSize, vocabSize, nLayers)

            # this maxVals version is more numerically stable, but I don't think the values are going to be very large because of the sigmoid in getPred
            # maxVals = np.max(preds, -1).reshape(-1, 1)
            # preds: np.ndarray = np.exp(preds - maxVals)
            preds: np.ndarray = np.exp(preds)

            # Get loss
            predsSum = preds.sum(axis=-1)
            tokenPreds = preds[indices, currentTokens]
            lossVals = np.log(tokenPreds / predsSum)
            loss -= lossVals.sum()

            # Get next state
            state = self.getNextStateBatched(
                weights, state, currentTokens, hiddenSize, vocabSize, nLayers
            )

        return loss / numTokens

    def getLossAndAccuracy(self, weights, tokens, hiddenSize, vocabSize, nLayers):
        loss = self.getLossBatched(weights, tokens, hiddenSize, vocabSize, nLayers)
        accuracy = np.exp(-loss)
        return loss, accuracy

    def preprocess(self, weights: np.ndarray, tokens, hiddenSize, vocabSize, nLayers):
        state = weights[:hiddenSize]
        for token in tokens:
            state = self.getNextState(
                weights, state, token, hiddenSize, vocabSize, nLayers
            )
        return state

    def generate(
        self,
        weights,
        state,
        hiddenSize,
        vocabSize,
        nLayers,
        stopToken,
        maxNumTokens=None,
    ):
        if state is None:
            state = weights[:hiddenSize]
        tokens = []
        while True:
            tokProbs = softmax(
                self.getPred(weights, state, hiddenSize, vocabSize, nLayers)
            )
            token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
            if token == stopToken:
                return tokens
            tokens.append(token)
            state = self.getNextState(
                weights, state, token, hiddenSize, vocabSize, nLayers
            )
            if maxNumTokens != None and len(tokens) == maxNumTokens:
                return tokens

    def getWeights(self, hiddenSize, vocabSize, nLayers):
        return (
            np.random.randn(
                hiddenSize  # initial state
                + nLayers
                * (
                    hiddenSize * hiddenSize + hiddenSize * vocabSize
                )  # hh and ih for each layer
                + hiddenSize * (hiddenSize * 4)  # ho1
                + (hiddenSize * 4) * vocabSize  # ho2
            ).astype(np.float32)
            * 0.02
        )


class AdamOptimizer:
    def __init__(self, nParams, alpha, beta1=0.9, beta2=0.999, eps=1e-5) -> None:
        self.nParams = nParams
        self.alpha = alpha
        self.beta1 = beta1
        self.beta1Power = beta1
        self.beta2 = beta2
        self.beta2Power = beta2
        self.eps = eps
        self.t = 0

        self.m = np.zeros(nParams)
        self.v = np.zeros(nParams)

    def getGrad(self, grad):
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad
        grad = (self.alpha * self.m * (1.0 / (1.0 - self.beta1Power))) / (
            np.sqrt(self.v * (1.0 / (1.0 - self.beta2Power))) + self.eps
        )
        self.beta1Power *= self.beta1
        self.beta2Power *= self.beta2
        self.t += 1
        return grad


def start_client():
    print("Started client")
    # Server settings
    server_ip = "130.215.211.30"
    # server_ip = "10.0.0.239"
    server_port = 55551

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
    # receive weights
    print("Waiting to receive weights")
    weights, valid = receiveData(client, "np.float32", "SERVER")
    weights = weights.copy()
    grad: np.ndarray = np.zeros_like(weights)
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
    maxNumTokens = float(max(tokenInfo))
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
        elapsed = perf_counter() - start
        print(f"{elapsed}s")

        # Receieve new tokens
        # receive tokens
        print("Waiting to receive tokens ", end="")
        start = perf_counter()
        tokens, valid = receiveData(client, "np.uint16", "SERVER")
        elapsed = perf_counter() - start
        print(f"{elapsed}s")
        # receive tokens info
        print("Waiting to receive token info ", end="")
        start = perf_counter()
        tokenInfo, valid = receiveData(client, "pickle", "SERVER")
        elapsed = perf_counter() - start
        print(f"{elapsed}s")
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
        trialStart = perf_counter()
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
        data, valid = receiveData(client, "pickle", "SERVER")
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
        grad *= 1.0 / float(totalNTrials)
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
    start_client()
