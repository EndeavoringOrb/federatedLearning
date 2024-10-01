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


def getLoss(weights, tokens, hiddenSize, vocabSize):
    state = weights[:hiddenSize]
    loss = 0.0
    for token in tokens:
        token = token.astype(np.uint32)
        tokProbs = softmax(
            state
            @ weights[
                hiddenSize + hiddenSize * hiddenSize + hiddenSize * vocabSize :
            ].reshape(hiddenSize, vocabSize)
        )
        loss -= np.log(tokProbs[token])
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
    return loss / len(tokens)


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
        grad = self.alpha * self.m * (1.0 / (1.0 - self.beta1Power)) / (np.sqrt(self.v * (1.0 / (1.0 - self.beta2Power))) + self.eps)
        self.beta1Power *= self.beta1
        self.beta2Power *= self.beta2
        self.t += 1
        return grad