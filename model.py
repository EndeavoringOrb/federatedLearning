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
