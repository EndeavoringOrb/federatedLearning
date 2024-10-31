import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

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

    def getInitState(self):
        return self.initState

    def forward(self, state, token):
        for i in range(self.nLayers):
            state = state + (state @ self.hh[i]) * self.ih[i][token]
            state = torch.tanh(state)
        return state

    def out(self, state):
        return state @ self.ho


if __name__ == "__main__":
    # SETTINGS
    ##########
    hiddenSize = 16
    vocabSize = 76
    nLayers = 4

    batchSize = 2
    learningRate = 1e-4
    ##########

    print(f"Loading data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenGenerator = tokenLoader(vocabSize, batchSize)

    print(f"Initializing model")
    model = SimpleRNN(vocabSize, hiddenSize, nLayers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    stepNum = 0

    print(f"Training")
    tokens, batchInfo = next(tokenGenerator)
    batchTokens = []
    for length in batchInfo:
        batchTokens.append(tokens[:length])
        tokens = tokens[length:]
    while True:
        batchTokens = [
            torch.tensor(sequence, dtype=torch.int64, device=device)
            for sequence in batchTokens
        ]

        loss = 0.0
        batchNumTokens = sum(batchInfo)

        for sequence in batchTokens:
            state = model.getInitState().to(device)

            for token in tqdm(sequence):
                out = model.out(state)
                loss += criterion(out, token)
                state = model(state, token)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stepNum += 1

        print(f"Step: {stepNum:,}, Loss: {loss / batchNumTokens}")
