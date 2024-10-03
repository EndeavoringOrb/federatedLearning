import numpy as np


def softmax(x):
    x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    invSum = 1.0 / np.sum(x)
    return x * invSum


def getWeights(hiddenSize, vocabSize):
    return (
        np.random.randn(
            hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize * vocabSize
        ).astype(np.float32)
        * 0.02
    )


def getPred(weights, state, hiddenSize, vocabSize):
    return state @ weights[
        hiddenSize + hiddenSize * hiddenSize + hiddenSize * vocabSize :
    ].reshape(hiddenSize, vocabSize)


def getNextState(weights, state, token, hiddenSize, vocabSize):
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


def getLoss(weights, tokens, hiddenSize, vocabSize):
    state = weights[:hiddenSize]
    loss = 0.0
    for token in tokens:
        token = token.astype(np.uint32)
        preds = getPred(weights, state, hiddenSize, vocabSize)
        preds = np.exp(preds - np.max(preds))
        loss -= np.log(preds[token] / np.sum(preds))
        state = getNextState(weights, state, token, hiddenSize, vocabSize)
    return loss / len(tokens)


def getAccuracy(weights, tokens, hiddenSize, vocabSize):
    state = weights[:hiddenSize]
    correct = 0.0
    for token in tokens:
        token = token.astype(np.uint32)
        preds = getPred(weights, state, hiddenSize, vocabSize)
        correct += (np.argmax(preds) == token)
        state = getNextState(weights, state, token, hiddenSize, vocabSize)
    return correct / len(tokens)


def generate(weights, tokens, hiddenSize, vocabSize, stopToken, maxNumTokens=None):
    state = weights[:hiddenSize]
    for token in tokens:
        state = getNextState(weights, state, token, hiddenSize, vocabSize)
    if maxNumTokens == None:
        while True:
            tokProbs = softmax(getPred(weights, state, hiddenSize, vocabSize))
            token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
            if token == stopToken:
                return tokens
            tokens.append(token)
            state = getNextState(weights, state, token, hiddenSize, vocabSize)
    else:
        for i in range(maxNumTokens):
            tokProbs = softmax(getPred(weights, state, hiddenSize, vocabSize))
            token = np.random.choice(vocabSize, 1, True, tokProbs)[0]
            if token == stopToken:
                return tokens
            tokens.append(token)
            state = getNextState(weights, state, token, hiddenSize, vocabSize)
    return tokens


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
            self.alpha
            * self.m
            * (1.0 / (1.0 - self.beta1Power))
            / (np.sqrt(self.v * (1.0 / (1.0 - self.beta2Power))) + self.eps)
        )
        self.beta1Power *= self.beta1
        self.beta2Power *= self.beta2
        self.t += 1
        return grad
