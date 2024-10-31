from trainNormal import SimpleRNN
from shakespeareData.shakespeareData import tokenLoader, Tokenizer
from time import perf_counter
import matplotlib.pyplot as plt
import torch

# Load model and tokenizer
print("Loading model")
model: SimpleRNN = torch.load("model.pt", weights_only=False).to("cpu")
tokenizer = Tokenizer(model.vocabSize)

print(f"Hidden Size: {model.hiddenSize}")
print(f"Vocab Size: {model.vocabSize}")
print(f"# Layers: {model.nLayers}")
print(f"# Parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"Vocab:")
print(tokenizer.chars)

while True:
    text = input("\nEnter text: ")
    start = perf_counter()
    tokens = tokenizer.tokenize(text)
    end = perf_counter()
    numInputTokens = len(tokens)
    print(
        f"Tokenized {numInputTokens} tokens in {1000*(end-start):.3f}ms ({int(numInputTokens/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    state = model.preprocess(tokens)
    end = perf_counter()
    print(
        f"Preprocessed {len(tokens)} tokens in {1000*(end-start):.3f}ms ({int(len(tokens)/(end-start)):,} tok/sec)"
    )

    start = perf_counter()
    newTokens = model.generate(state, tokenizer.stopToken)
    end = perf_counter()
    print(
        f"Generated {len(newTokens)} tokens in {1000*(end-start):.3f}ms ({int(len(newTokens)/(end-start)):,} tok/sec)"
    )

    print()
    print(tokenizer.deTokenize(tokens + newTokens))