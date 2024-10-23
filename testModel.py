from utilities.model import *
from shakespeareData.shakespeareData import tokenLoader, Tokenizer
from time import perf_counter
import matplotlib.pyplot as plt
import json


def testChatModel():
    text = input("Enter text: ")
    start = perf_counter()
    tokens = tokenizer.tokenize(text)
    end = perf_counter()
    numInputTokens = len(tokens)
    print(
        f"Tokenized {numInputTokens} tokens in {1000*(end-start):.3f}ms ({int(numInputTokens/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    state = model.preprocess(weights, tokens, hiddenSize, vocabSize, nLayers)
    end = perf_counter()
    print(
        f"Preprocessed {len(tokens)} tokens in {1000*(end-start):.3f}ms ({int(len(tokens)/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    newTokens = model.generate(
        weights, state, hiddenSize, vocabSize, nLayers, tokenizer.stopToken
    )
    end = perf_counter()
    print(
        f"Generated {len(newTokens)} tokens in {1000*(end-start):.3f}ms ({int(len(newTokens)/(end-start)):,} tok/sec)"
    )

    print()
    print(tokenizer.deTokenize(tokens + newTokens))
    print()


runNum = int(input("Enter the run # you would like to test: "))
folder = f"trainingRuns/{runNum}"

with open(f"{folder}/loss.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()
lossValues = [float(item.strip()) for item in text.split("\n")]

plt.plot(lossValues)
plt.ylabel("Loss Value")
plt.xlabel("Training Step")
plt.savefig(f"{folder}/loss.svg")
plt.show()

with open(f"{folder}/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

weights = np.load(f"{folder}/model.npy")

hiddenSize = config["hiddenSize"]
vocabSize = config["vocabSize"]
nLayers = config["nLayers"]
modelType = config["modelType"]
weights = weights[3:]

if config["modelType"] == "chat":
    model = ChatModel()
else:
    model = ChatModel()

tokenizer = Tokenizer(vocabSize)

print(f"Loaded model")
print(f"Hidden Size: {hiddenSize}")
print(f"Vocab Size: {vocabSize}")
print(f"# Layers: {nLayers}")
print(f"# Parameters: {weights.shape[0]:,}")

print(f"Vocab:")
print(tokenizer.chars)

print("Loading tokens")
tokens, tokenInfo = next(tokenLoader(vocabSize, 512))
batchTokens = []
for length in tokenInfo:
    batchTokens.append(tokens[:length])
    tokens = tokens[length:]
totalNumTokens = sum(tokenInfo)

print("De-Tokenize Test:")
print(tokenizer.deTokenize(batchTokens[0]) + "\n")

print(f"Testing {len(tokenInfo):,} random chunks ({sum(tokenInfo):,} tokens)")
loss, accuracy = model.getLossAndAccuracy(
    weights, batchTokens, hiddenSize, vocabSize, nLayers
)
print(f"Model Avg. Loss: {loss:.3e}")
print(f"Model Accuracy: {100*accuracy:.3f}%")

while True:
    if modelType == "chat":
        testChatModel()
    else:
        testChatModel()
