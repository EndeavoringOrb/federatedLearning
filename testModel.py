from utilitiesModel import *
from utilitiesData import *
from time import perf_counter

modelType = "critic"

if modelType == "critic":
    model = ChatCritic()
else:
    model = ChatModel()

def testChatCritic():
    text = input("Enter text: ")
    start = perf_counter()
    tokens = tokenize(text)
    if len(tokens) == 0 or tokens[-1] != stopToken:
        tokens.append(stopToken)
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
    tokens = tokenize(text)
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

    start = perf_counter()
    newTokens = model.generate(weights, state, hiddenSize, vocabSize, tokenize("|"))
    end = perf_counter()
    print(
        f"Generated {len(newTokens)} tokens in {1000*(end-start):.3f}ms ({int(len(newTokens)/(end-start)):,} tok/sec)"
    )

    print()
    print(deTokenize(tokens + newTokens))
    print()



weights = np.load("weights/model.npy")

hiddenSize = int(weights[0])
vocabSize = int(weights[1])
weights = weights[2:]

print(f"Loaded model")
print(f"Hidden Size: {hiddenSize}")
print(f"Vocab Size: {vocabSize}")
print(f"# Parameters: {weights.shape[0]}")

print(f"Vocab:")
print(chars)

tokens = loadAllChatTokens()
print(f"Testing {len(tokens)} chunks")
loss, accuracy = model.getLossAndAccuracy(weights, [[chunk, 1] for chunk in tokens], hiddenSize, vocabSize)
print(f"Model Avg. Loss: {loss:.3e}")
print(f"Model Accuracy: {100*accuracy:.3f}%")

while True:
    if modelType == "critic":
        testChatCritic()
    else:
        testChatModel()