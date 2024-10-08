from utilitiesModel import *
from redditData import tokenLoader, Tokenizer
from time import perf_counter

modelType = "chat"

if modelType == "critic":
    model = ChatCritic()
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
    state = model.preprocess(weights, tokens, hiddenSize, vocabSize)
    end = perf_counter()
    print(
        f"Preprocessed {len(tokens)} tokens in {1000*(end-start):.3f}ms ({int(len(tokens)/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    newTokens = model.generate(weights, state, hiddenSize, vocabSize, tokenizer.stopToken)
    end = perf_counter()
    print(
        f"Generated {len(newTokens)} tokens in {1000*(end-start):.3f}ms ({int(len(newTokens)/(end-start)):,} tok/sec)"
    )

    print()
    print(tokenizer.deTokenize(tokens + newTokens))
    print()


weights = np.load("weights/model.npy")

hiddenSize = int(weights[0])
vocabSize = int(weights[1])
weights = weights[2:]

tokenizer = Tokenizer(vocabSize)

print(f"Loaded model")
print(f"Hidden Size: {hiddenSize}")
print(f"Vocab Size: {vocabSize}")
print(f"# Parameters: {weights.shape[0]}")

print(f"Vocab:")
print(tokenizer.chars)

tokens = []
for i, item in enumerate(tokenLoader(vocabSize, False)):
    tokens.append(item)
    if i == 99:
        break
print(f"Testing {len(tokens):,} chunks ({sum([len(chunk) for chunk in tokens]):,} tokens)")
loss, accuracy = model.getLossAndAccuracy(
    weights, tokens, hiddenSize, vocabSize
)
print(f"Model Avg. Loss: {loss:.3e}")
print(f"Model Accuracy: {100*accuracy:.3f}%")

while True:
    if modelType == "critic":
        testChatCritic()
    else:
        testChatModel()
