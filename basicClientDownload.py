import socket
import struct
import pickle
import numpy as np
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
            raise Exception('unexpected EOF')
        received_chunks.append(received)
        remaining -= len(received)
    return b''.join(received_chunks)

def sendBytes(connection: socket.socket, data: bytes, addr):
    log(f"SENDING DATA")
    dataLength = len(data)

    # send header
    connection.send(struct.pack(headerFormat, dataLength))
    log(f"  Sent header specifying length of {len(data)} bytes")
    msg = recvall(connection, headerSize)
    dataLengthEcho = struct.unpack(headerFormat, msg)[0]

    if dataLength != dataLengthEcho:
        log(f"  Header not correctly received. Resending header.")

    while dataLength != dataLengthEcho:
        connection.send(struct.pack(headerFormat, dataLength))
        msg = recvall(connection, headerSize)
        dataLengthEcho = struct.unpack(headerFormat, msg)[0]

    connection.send(
        struct.pack(headerFormat, 0)
    )  # send message to confirm it was correctly received
    log(f"  Sent validation header")

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

    # see if chunks were properly received
    msg = recvall(connection, headerSize)
    msgLength = struct.unpack(headerFormat, msg)[0]
    msg = recvall(connection, msgLength)
    properlyReceived = pickle.loads(msg) == "chunksGood"
    if properlyReceived:
        log(f"  Chunks were properly received")
    else:
        log(f"  Chunks were NOT properly received")

    # re-send stuff if not properly received
    while not properlyReceived:
        # receive list of failed chunks
        msg = recvall(connection, headerSize)
        msgLength = struct.unpack(headerFormat, msg)[0]
        msg = recvall(connection, msgLength)
        failedChunks = pickle.loads(msg)
        log(f"  Got list of failed chunks {failedChunks}")

        # re-send failed chunks
        log(f"  Re-sending failed chunks")
        for chunkNum in failedChunks:
            connection.send(dataChunks[chunkNum])
            log(f"  Re-sent {len(dataChunks[chunkNum])} bytes")

        # see if chunks were properly received
        msg = recvall(connection, headerSize)
        msgLength = struct.unpack(headerFormat, msg)[0]
        msg = recvall(connection, msgLength)
        properlyReceived = pickle.loads(msg) == "chunksGood"

        if properlyReceived:
            log(f"  Chunks were properly received")
        else:
            log(f"  Chunks were NOT properly received")


def receiveData(connection: socket.socket, dataType: str, addr):
    log("RECEIVING DATA")
    msg = recvall(connection, headerSize)
    msgLength = struct.unpack(headerFormat, msg)[0]
    if not msg:
        log(f"  Received empty message from {addr}")
        return msg, False

    log(f"  Sending header echo")
    connection.send(struct.pack(headerFormat, msgLength))
    msg = recvall(connection, headerSize)
    echoLength = struct.unpack(headerFormat, msg)[0]
    log(f"  Received header echo {echoLength}")

    while echoLength != 0:
        log(f"  Header verification failed. Re-requesting header.")
        msg = recvall(connection, headerSize)
        echoLength = struct.unpack(headerFormat, msg)[0]
        log(f"  Received header echo {echoLength}")
        if echoLength == 0:
            break
        msgLength = echoLength
        connection.send(struct.pack(headerFormat, echoLength))

    log(f"  Receiving message with length {msgLength}")

    # Receive message bytes in chunks
    messages = []
    numChunks = (msgLength + BUFFER_SIZE - 1) // BUFFER_SIZE
    remainderBytes = msgLength - (msgLength // BUFFER_SIZE) * BUFFER_SIZE
    log(f"  Remainder bytes: {remainderBytes}")
    for i in range(numChunks):
        if i == numChunks - 1 and remainderBytes > 0:
            chunkLength = remainderBytes
        else:
            chunkLength = BUFFER_SIZE
        msg = recvall(connection, chunkLength)
        messages.append(msg)
        log(
            f"  Received chunk [{i+1}/{numChunks}] with length {len(msg)}/{chunkLength}"
        )

    # Check for any chunks that were not received fully
    log(f"  Validating chunks were fully received")
    failedChunks = []
    for i in range(numChunks - 1):
        if len(messages[i]) != BUFFER_SIZE:
            failedChunks.append((i, BUFFER_SIZE))
    if remainderBytes > 0 and len(messages[-1]) != remainderBytes:
        failedChunks.append((numChunks - 1, remainderBytes))

    # Send a message to say if all the data was properly received
    data = pickle.dumps("chunksBad" if len(failedChunks) > 0 else "chunksGood")
    dataLength = len(data)
    connection.send(struct.pack(headerFormat, dataLength))
    connection.send(data)

    while len(failedChunks) > 0:
        log(
            f"  {len(failedChunks)} chunks not fully received, re-requesting chunks {failedChunks}"
        )
        # Re-request any chunks that failed
        data = pickle.dumps(
            [item[0] for item in failedChunks]
        )  # send the indices of the chunks that failed
        dataLength = len(data)
        connection.send(struct.pack(headerFormat, dataLength))
        connection.send(data)

        # Receive those chunks
        for i, (chunkNum, chunkLength) in enumerate(failedChunks):
            msg = recvall(connection, chunkLength)
            messages[chunkNum] = msg
            log(f"  Received re-requested chunk [{i+1}/{len(failedChunks)}] ({len(msg)} bytes)")

        # Check for any chunks that were not received fully
        log(f"  Validating chunks were fully received")
        failedChunks = []
        for i in range(numChunks - 1):
            if len(messages[i]) != BUFFER_SIZE:
                failedChunks.append((i, BUFFER_SIZE))
        if remainderBytes > 0 and len(messages[-1]) != remainderBytes:
            failedChunks.append((numChunks - 1, remainderBytes))

        # Send a message to say if all the data was properly received
        data = pickle.dumps("chunksBad" if len(failedChunks) > 0 else "chunksGood")
        dataLength = len(data)
        connection.send(struct.pack(headerFormat, dataLength))
        connection.send(data)

    # Concatenate all messages
    msg = b""
    for message in messages:
        msg += message

    log(f"  Received {len(msg)}/{msgLength} bytes in {numChunks} chunks")

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


class ChatModel:
    def __init__(self) -> None:
        pass

    def getPred(self, weights, state, hiddenSize, vocabSize):
        return state @ weights[
            hiddenSize + hiddenSize * hiddenSize + hiddenSize * vocabSize :
        ].reshape(hiddenSize, vocabSize)

    def getNextState(self, weights, state, token, hiddenSize, vocabSize):
        state = np.tanh(
            state
            + state
            @ weights[hiddenSize : hiddenSize + hiddenSize * hiddenSize].reshape(
                hiddenSize, hiddenSize
            )
            + weights[
                hiddenSize
                + hiddenSize * hiddenSize
                + hiddenSize * token : hiddenSize
                + hiddenSize * hiddenSize
                + hiddenSize * (token + 1)
            ]
        )
        return state

    def getLoss(self, weights, tokens, hiddenSize, vocabSize):
        loss = 0.0
        numTokens = 0

        for chunk in tokens:
            state = weights[:hiddenSize]
            numTokens += len(chunk)

            for token in chunk:
                token = token.astype(np.uint32)
                preds = self.getPred(weights, state, hiddenSize, vocabSize)
                preds = np.exp(preds - np.max(preds))
                loss -= np.log(preds[token] / np.sum(preds))
                state = self.getNextState(weights, state, token, hiddenSize, vocabSize)

        return loss / numTokens

    def getAccuracy(self, weights, tokens, hiddenSize, vocabSize):
        state = weights[:hiddenSize]
        correct = 0.0
        numTokens = 0
        for chunk in tokens:
            numTokens += len(chunk)
            for token in chunk:
                token = token.astype(np.uint32)
                preds = self.getPred(weights, state, hiddenSize, vocabSize)
                correct += np.argmax(preds) == token
                state = self.getNextState(weights, state, token, hiddenSize, vocabSize)
        return correct / numTokens

    def getLossAndAccuracy(self, weights, tokens, hiddenSize, vocabSize):
        loss = 0.0
        accuracy = 0.0
        numTokens = 0

        for chunk in tokens:
            state = weights[:hiddenSize]
            numTokens += len(chunk)

            for token in chunk:
                token = token.astype(np.uint32)
                preds = self.getPred(weights, state, hiddenSize, vocabSize)
                accuracy += np.argmax(preds) == token
                preds = np.exp(preds - np.max(preds))
                loss -= np.log(preds[token] / np.sum(preds))
                state = self.getNextState(weights, state, token, hiddenSize, vocabSize)

        return loss / numTokens, accuracy / numTokens

    def preprocess(self, weights, tokens, hiddenSize, vocabSize):
        state = weights[:hiddenSize]
        for token in tokens:
            state = self.getNextState(weights, state, token, hiddenSize, vocabSize)
        return state

    def generate(
        self, weights, state, hiddenSize, vocabSize, stopToken, maxNumTokens=None
    ):
        if state is None:
            state = weights[:hiddenSize]
        tokens = []
        if maxNumTokens == None:
            while True:
                tokProbs = softmax(self.getPred(weights, state, hiddenSize, vocabSize))
                token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
                if token == stopToken:
                    return tokens
                tokens.append(token)
                state = self.getNextState(weights, state, token, hiddenSize, vocabSize)
        else:
            for i in range(maxNumTokens):
                tokProbs = softmax(self.getPred(weights, state, hiddenSize, vocabSize))
                token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
                if token == stopToken:
                    return tokens
                tokens.append(token)
                state = self.getNextState(weights, state, token, hiddenSize, vocabSize)
        return tokens

    def getWeights(self, hiddenSize, vocabSize):
        return (
            np.random.randn(
                hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize * vocabSize
            ).astype(np.float32)
            * 0.02
        )


class ChatCritic:
    def __init__(self) -> None:
        pass

    def getNextState(self, weights, state, token, hiddenSize, vocabSize):
        state = np.tanh(
            state
            + state
            @ weights[hiddenSize : hiddenSize + hiddenSize * hiddenSize].reshape(
                hiddenSize, hiddenSize
            )
            + weights[
                hiddenSize
                + hiddenSize * hiddenSize
                + hiddenSize * token : hiddenSize
                + hiddenSize * hiddenSize
                + hiddenSize * (token + 1)
            ]
        )
        return state

    def preprocess(self, weights, tokens, hiddenSize, vocabSize):
        state = weights[:hiddenSize]
        for token in tokens:
            state = self.getNextState(weights, state, token, hiddenSize, vocabSize)
        return state

    def getPred(self, weights, state, hiddenSize, vocabSize):
        """
        State -> [no, yes]
        """
        return state @ weights[
            hiddenSize + hiddenSize * hiddenSize + hiddenSize * vocabSize :
        ].reshape(hiddenSize, 2)
    
    def getLoss(self, weights, tokens, hiddenSize, vocabSize):
        loss = 0.0

        for chunk, goodAnswer in tokens:
            state = self.preprocess(weights, chunk, hiddenSize, vocabSize)
            preds = self.getPred(weights, state, hiddenSize, vocabSize)
            preds = np.exp(preds - np.max(preds))
            loss -= np.log(preds[goodAnswer] / np.sum(preds))

        return loss / len(tokens)

    def getLossAndAccuracy(self, weights, tokens, hiddenSize, vocabSize):
        loss = 0.0
        accuracy = 0.0

        for chunk, goodAnswer in tokens:
            state = self.preprocess(weights, chunk, hiddenSize, vocabSize)
            preds = self.getPred(weights, state, hiddenSize, vocabSize)
            accuracy += np.argmax(preds) == goodAnswer
            preds = np.exp(preds - np.max(preds))
            loss -= np.log(preds[goodAnswer] / np.sum(preds))

        return loss / len(tokens), accuracy / len(tokens)

    def getWeights(self, hiddenSize, vocabSize):
        return (
            np.random.randn(
                hiddenSize
                + hiddenSize * hiddenSize
                + hiddenSize * vocabSize
                + hiddenSize * 2
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
        grad = (
            (self.alpha
            * self.m
            * (1.0 / (1.0 - self.beta1Power)))
            / (np.sqrt(self.v * (1.0 / (1.0 - self.beta2Power))) + self.eps)
        )
        self.beta1Power *= self.beta1
        self.beta2Power *= self.beta2
        self.t += 1
        return grad

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
        print(f"Running trials for {config['timePerStep']}s on {len(tokens)} tokens")
        start = perf_counter()
        rewards = [seed]
        numTrials = 0
        np.random.seed(seed)
        while perf_counter() - start < config["timePerStep"]:
            loss = model.getLoss(
                weights + np.random.randn(weights.shape[0]) * config["sigma"],
                [tokens],
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

        reward_info = data["reward_info"]
        seed = data["seed"]

        # Update weights
        print(f"Updating weights")
        rewardNum = 0
        grad.fill(0)
        for nTrials, trialSeed in reward_info:
            np.random.seed(trialSeed)
            mulVal = config["sigma"] / float(nTrials) # normalize the grad by the number of samples per example
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
