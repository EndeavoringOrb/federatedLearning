import numpy as np
import json
import os


def textLoader():
    filenames = os.listdir("shakespeareData/works")
    for filename in filenames:
        # Open and read the file
        with open(f"shakespeareData/works/{filename}", "r") as f:
            text = f.read()

        # get chunks
        chunks = text.split("\n\n")

        # clean chunks
        chunks = [
            chunk.strip() for chunk in chunks if "==" not in chunk
        ]  # clean out stuff like "ACT 1\n====="

        for chunk in chunks:
            yield chunk

    with open(f"shakespeareData/sonnets.txt", "r") as f:
        text = f.read()

    # get chunks
    chunks = text.split("\n\n")

    # clean chunks
    chunks = [chunk.strip() for chunk in chunks if len(chunk) > 3]

    for chunk in chunks:
        yield chunk


class Tokenizer:
    def __init__(self, vocabSize) -> None:
        with open("shakespeareData/shakespeareFreqs.json", "r", encoding="utf-8") as f:
            characterFrequencies = json.load(f)
        assert (
            vocabSize <= len(characterFrequencies) + 2
        ), f"vocabSize ({vocabSize}) must be less than # characters ({len(characterFrequencies) + 2})"
        self.vocabSize = vocabSize
        self.chars = [item[0] for item in characterFrequencies[: vocabSize - 2]]
        self.itos = {i + 2: character for i, character in enumerate(self.chars)}
        self.stoi = {character: i + 2 for i, character in enumerate(self.chars)}
        self.unknownToken = 0
        self.stopToken = 1

    def tokenize(self, text, addStopToken=False):
        tokens = []
        for character in text:
            if character in self.chars:
                tokens.append(self.stoi[character])
            else:
                tokens.append(0)  # unknown token
        if addStopToken:
            tokens.append(self.stopToken)
        return tokens

    def deTokenize(self, tokens):
        text = ""
        for token in tokens:
            if token == self.unknownToken or token >= self.vocabSize:
                text += "<UNK>"
            elif token == self.stopToken:
                text += "<STOP>"
            else:
                text += self.itos[token]
        return text


def tokenLoader(vocabSize, batchSize):
    tokenizer = Tokenizer(vocabSize)

    tokens = [
        np.array(tokenizer.tokenize(text, True)).astype(np.uint16)
        for text in textLoader()
    ]

    # TODO: move this block back into the while True
    indices = np.arange(min(len(tokens), batchSize))
    #indices = np.random.choice(len(tokens), (batchSize,))
    batchTokens = []
    batchInfo = []
    for idx in indices:
        idxTokens = tokens[idx]
        batchTokens.append(idxTokens)
        batchInfo.append(len(idxTokens))
    batchTokens = np.concatenate(batchTokens)

    while True:
        yield batchTokens, batchInfo
