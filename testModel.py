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

print(chars)
print(tokenize("|"))

while True:
    text = input("Enter text: ")
    tokens = tokenize(text)
    tokens = generate(weights, tokens, hiddenSize, vocabSize, 0)
    print(decode(tokens))