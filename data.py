import numpy as np

with open("tokens.txt", "r", encoding="utf-8") as f:
    tokenText = f.read()

chars = set(tokenText)
chars = ["|"] + sorted(list(chars))
vocabSize = len(chars)
stoi = {character: idx for idx, character in enumerate(chars)}
itos = {idx: character for idx, character in enumerate(chars)}


def tokenize(text):
    return [stoi[character] for character in text]


def decode(tokens):
    return "".join([itos[token] for token in tokens])


def loadTokens():
    tokens = tokenize(tokenText)
    tokens = np.array(tokens, dtype=np.uint8)
    return tokens
