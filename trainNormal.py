import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from time import perf_counter

from shakespeareData.shakespeareData import tokenLoader


class SimpleRNN(nn.Module):
    def __init__(self, vocabSize, hiddenSize, nLayers):
        super(SimpleRNN, self).__init__()

        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.nLayers = nLayers

        data = torch.zeros((hiddenSize,))
        self.initState = nn.Parameter(data)

        data = torch.randn((nLayers, vocabSize, hiddenSize)) * 0.02
        self.ih = nn.Parameter(data)

        data = torch.randn((nLayers, hiddenSize, hiddenSize)) * 0.02
        self.hh = nn.Parameter(data)

        data = torch.randn((hiddenSize, vocabSize)) * 0.02
        self.ho = nn.Parameter(data)

    def getInitState(self, batchSize):
        return self.initState.unsqueeze(0).expand((batchSize, -1))

    def forward(self, state, token):
        for i in range(self.nLayers):
            state = state + (state @ self.hh[i]) * self.ih[i][token]
            state = torch.tanh(state)
        return state

    def out(self, state):
        return state @ self.ho

    @torch.no_grad
    def preprocess(self, tokens):
        state = self.getInitState(1)
        for token in tokens:
            state = self.forward(state, token)
        return state

    @torch.no_grad
    def generate(self, state, stopToken):
        newTokens = []
        while True:
            out = self.out(state)
            probs = torch.softmax(out, dim=-1)
            newToken = torch.multinomial(probs, 1)[0].item()
            if newToken == stopToken:
                break
            newTokens.append(newToken)
            state = self.forward(state, newToken)
        return newTokens


if __name__ == "__main__":
    # SETTINGS
    ##########
    hiddenSize = 32
    vocabSize = 76
    nLayers = 4

    batchSize = 16
    learningRate = 1e-4

    saveInterval = 10
    ##########

    print(f"Loading data")
    tokenGenerator = tokenLoader(vocabSize, batchSize)

    print(f"Initializing model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using {device}")
    # model: SimpleRNN = SimpleRNN(vocabSize, hiddenSize, nLayers).to(device)
    model: SimpleRNN = torch.load("model.pt", weights_only=False).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    stepNum = 0

    print(f"Training")
    while True:
        # Get tokens
        tokens, batchInfo = next(tokenGenerator)
        batchTokens = []
        for length in batchInfo:
            batchTokens.append(tokens[:length])
            tokens = tokens[length:]
        batchTokens = [
            torch.tensor(sequence, dtype=torch.int64, device=device)
            for sequence in batchTokens
        ]

        # Batch forward pass
        loss = 0.0
        batchNumTokens = sum(batchInfo)
        chunkLengths = torch.tensor(batchInfo, dtype=torch.int64)
        sortedChunkLengths = sorted(list(set(batchInfo)))
        indices = torch.arange(batchSize)
        state = model.getInitState(batchSize).to(device)

        start = perf_counter()  # Start timer

        previousLength = 0
        for length in sortedChunkLengths:
            # Remove sequences that are finished
            # TODO: optimize this by doing a nested for loop with the chunkLengths
            removeIndices = torch.where(chunkLengths == previousLength)[0]
            notRemoveIndices = torch.where(chunkLengths != previousLength)[0]
            if len(removeIndices) > 0:
                removeIndices, _ = torch.sort(removeIndices, descending=True)
                for idx in removeIndices:
                    batchTokens.pop(idx)
                chunkLengths = chunkLengths[notRemoveIndices]
                state = state[notRemoveIndices]
                indices = np.arange(len(chunkLengths))

            for i in range(previousLength, length):
                # Get tokens for current step
                currentTokens = torch.tensor(
                    [chunk[i] for chunk in batchTokens],
                    dtype=torch.int64,
                    device=device,
                )

                # Get preds
                out = model.out(state)

                # Get loss
                loss += criterion(out, currentTokens)

                # Get next state
                state = model(state, currentTokens)

            previousLength = length

        """for sequence in batchTokens:
            state = model.getInitState().to(device)

            for token in sequence:
                out = model.out(state)
                loss += criterion(out, token)
                state = model(state, token)"""

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = perf_counter() - start

        # Tracking
        stepNum += 1

        # Logging
        if stepNum % saveInterval == 0:
            torch.save(model, "model.pt")
        print(
            f"Step: {stepNum:,}, Loss: {loss / max(batchInfo):.4e}, {int(batchNumTokens / elapsed):,} tok/sec"
        )
