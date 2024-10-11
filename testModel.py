from utilitiesModel import *
from shakespeareData import tokenLoader, Tokenizer
from time import perf_counter
import matplotlib.pyplot as plt

modelType = "chat"

if modelType == "critic":
    model = ChatCritic()
elif modelType == "minGru":
    model = MinGruChat()
else:
    model = ChatModel()


def testChatCritic():
    text = input("Enter text: ")
    start = perf_counter()
    tokens = tokenizer.tokenize(text)
    if len(tokens) == 0 or tokens[-1] != tokenizer.stopToken:
        tokens.append(tokenizer.stopToken)
    end = perf_counter()
    numInputTokens = len(tokens)
    print(
        f"Tokenized {numInputTokens} tokens in {1000*(end-start):.3f}ms ({int(numInputTokens/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    state = model.preprocess(weights, tokens, hiddenSize, vocabSize)
    end = perf_counter()
    print(
        f"Preprocessed {len(tokens)} tokens in {1000*(end-start):.3f}ms ({int(len(tokens)/(end-start)):,} tok/sec)"
    )

    preds = model.getPred(weights, state, hiddenSize, vocabSize)
    preds = softmax(preds)

    print(f"Inputted text has an Answer Score of {100*preds[1]:.3f}%")
    print()


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


folder = "trainingRuns/25"

with open(f"{folder}/loss.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()
lossValues = [float(item.strip()) for item in text.split("\n")]

plt.plot(lossValues)
plt.show()

weights = np.load(f"{folder}/model.npy")

hiddenSize = int(weights[0])
vocabSize = int(weights[1])
nLayers = int(weights[2])
weights = weights[3:]

tokenizer = Tokenizer(vocabSize)

print(f"Loaded model")
print(f"Hidden Size: {hiddenSize}")
print(f"Vocab Size: {vocabSize}")
print(f"# Layers: {nLayers}")
print(f"# Parameters: {weights.shape[0]:,}")

print(f"Vocab:")
print(tokenizer.chars)

print("Loading tokens")
tokens, tokenInfo = next(tokenLoader(vocabSize, 8))
batchTokens = []
for length in tokenInfo:
    batchTokens.append(tokens[:length])
    tokens = tokens[length:]
totalNumTokens = sum(tokenInfo)

print(f"Testing {len(tokenInfo):,} random chunks ({sum(tokenInfo):,} tokens)")
loss, accuracy = model.getLossAndAccuracy(
    weights, batchTokens, hiddenSize, vocabSize, nLayers
)
print(f"Model Avg. Loss: {loss:.3e}")
print(f"Model Accuracy: {100*accuracy:.3f}%")

while True:
    if modelType == "critic":
        testChatCritic()
    else:
        testChatModel()
