from model import *
from data import *

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
print(tokenize("|"))

tokens = loadTokens()
print(f"Loaded {len(tokens)} for testing")
loss = getLoss(weights, tokens, hiddenSize, vocabSize)
print(f"Model Avg. Loss: {loss:.3e}")
accuracy = getAccuracy(weights, tokens, hiddenSize, vocabSize)
print(f"Model Accuracy: {100*accuracy:.3f}%")

while True:
    text = input("Enter text: ")
    tokens = tokenize(text)
    tokens = generate(weights, tokens, hiddenSize, vocabSize, tokenize("|"))
    print(deTokenize(tokens))