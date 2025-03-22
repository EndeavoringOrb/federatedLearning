import numpy as np


def softmax(x):
    np.subtract(x, np.max(x), x)  # Subtract max for numerical stability
    np.exp(x, x)
    invSum = 1.0 / np.sum(x)
    np.multiply(x, invSum, x)


def relu(x):
    np.maximum(x, 0, x)


class ChatModel:
    def __init__(self) -> None:
        pass

    def getInitState(self, weights: np.ndarray, hiddenSize):
        return weights[:hiddenSize].copy()

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

    def getLossBatched(
        self, weights, tokens: list[list], hiddenSize, vocabSize, nLayers
    ):
        loss = 0.0
        batchSize = len(tokens)

        state = (
            self.getInitState(weights, hiddenSize).reshape(1, -1).repeat(batchSize, 0)
        )
        chunkLengths = np.asarray([len(chunk) for chunk in tokens])
        sortedChunkLengths = np.asarray(
            sorted(list(set([len(chunk) for chunk in tokens])))
        )
        numTokens = sum(chunkLengths)
        indices = np.arange(batchSize)

        previousLength = 0
        for length in sortedChunkLengths:
            # Remove sequences that are finished
            # TODO: optimize this by doing a nested for loop with the chunkLengths
            removeIndices: np.ndarray = np.where(chunkLengths == previousLength)[0]
            if len(removeIndices) > 0:
                removeIndices[::-1].sort()
                for idx in removeIndices:
                    tokens.pop(idx)
                chunkLengths = np.delete(chunkLengths, removeIndices)
                state = np.delete(state, removeIndices, axis=0)
                indices = np.arange(len(chunkLengths))

            for i in range(previousLength, length):
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

            previousLength = length

        return loss / numTokens

    def getLossAndAccuracy(self, weights, tokens, hiddenSize, vocabSize, nLayers):
        loss = self.getLossBatched(weights, tokens, hiddenSize, vocabSize, nLayers)
        accuracy = np.exp(-loss)
        return loss, accuracy

    def preprocess(self, weights: np.ndarray, tokens, hiddenSize, vocabSize, nLayers):
        state = self.getInitState(weights, hiddenSize)
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
        numBeams=None,
    ):
        if numBeams == None:
            if state is None:
                state = self.getInitState(weights, hiddenSize)
            tokens = []
            while True:
                tokProbs = self.getPred(weights, state, hiddenSize, vocabSize, nLayers)
                softmax(tokProbs)
                token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
                if token == stopToken:
                    return tokens
                tokens.append(token)
                state = self.getNextState(
                    weights, state, token, hiddenSize, vocabSize, nLayers
                )
                if maxNumTokens != None and len(tokens) == maxNumTokens:
                    return tokens
        else:
            if state is None:
                state = self.getInitState(weights, hiddenSize)
            state = state.reshape(1, hiddenSize).repeat(numBeams, 0)
            candidate_seqs = [[[], i, 0] for i in range(numBeams)]
            stepNum = 0
            while True:
                next_candidate_seqs = []

                for candidate in candidate_seqs:
                    if len(candidate[0]) > 0 and candidate[0][-1] == stopToken:
                        next_candidate_seqs.append(candidate)
                        continue

                    tokProbs = self.getPred(
                        weights, state[candidate[1]], hiddenSize, vocabSize, nLayers
                    )
                    softmax(tokProbs)
                    tokens = np.random.choice(vocabSize, numBeams, False, tokProbs)

                    for i in range(numBeams):
                        nextTokId = tokens[i]
                        nextScore = tokProbs[nextTokId]
                        newSeq = [
                            [item for item in candidate[0]] + [nextTokId],
                            candidate[1],
                            candidate[2] + np.log(nextScore),
                        ]
                        next_candidate_seqs.append(newSeq)

                next_candidate_seqs.sort(key=lambda x: x[2], reverse=True)
                candidate_seqs = next_candidate_seqs[:numBeams]
                state = state[[item[1] for item in candidate_seqs]]

                for i in range(numBeams):
                    candidate_seqs[i][1] = i
                    state[i] = self.getNextState(
                        weights,
                        state[i],
                        candidate_seqs[i][0][-1],
                        hiddenSize,
                        vocabSize,
                        nLayers,
                    )

                stepNum += 1

                if (
                    all(seq[0][-1] == stopToken for seq in candidate_seqs)
                    or stepNum >= 1000
                ):
                    return candidate_seqs[0][0][:-1]

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
