import numpy as np
import json
import os


def textLoader():
    filenames = os.listdir("redditData")
    for filename in filenames:
        # Open and read the file
        with open(f"redditData/{filename}", "r") as file:
            for line in file:
                # Parse each line
                item = json.loads(line)
                title = item["title"]
                body = item["body"]
                answers = [answer["body"] for answer in item["answers"]]

                # Yield the examples
                for answer in answers:
                    question = f"Question: {title}\n\n{body}".strip() + "\n\n"
                    yield len(question), f"{question}{answer}"


class Tokenizer:
    def __init__(self, vocabSize) -> None:
        with open("redditFreqs.json", "r", encoding="utf-8") as f:
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


def tokenLoader(vocabSize, repeat):
    tokenizer = Tokenizer(vocabSize)

    while True:
        for questionLength, text in textLoader():
            yield np.array(tokenizer.tokenize(text, True)).astype(np.uint16)
        if not repeat:
            break