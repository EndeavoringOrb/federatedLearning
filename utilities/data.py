import numpy as np
import os


def chatTextLoader():
    filenames = sorted(os.listdir("chatData"), key=lambda x: int(x.split(".")[0]))
    for filename in filenames:
        try:
            with open(f"chatData/{filename}", "r", encoding="utf-8") as f:
                text = f.read()
                yield text
        except FileNotFoundError:
            pass


# Create vocabulary
chars = set()
for text in chatTextLoader():
    chars.update(text)
chars = sorted(list(chars))

# Create character maps
vocabSize = len(chars) + 6
stoi = {character: idx + 6 for idx, character in enumerate(chars)}
itos = {idx + 6: character for idx, character in enumerate(chars)}

stopToken = 0
userToken = 1
queryToken = 2
getSourcesToken = 3
readSourceToken = 4
respondToken = 5


def tokenize(text):
    return [stoi[character] for character in text]


def deTokenize(tokens):
    return "".join(
        [
            itos[token]
            for token in tokens
            if token
            not in [
                stopToken,
                userToken,
                queryToken,
                getSourcesToken,
                readSourceToken,
                respondToken,
            ]
        ]
    )


def loadAllChatTokens():
    tokens = [item for item in chatTokenLoader(False)]
    print(f"Loaded {sum([len(chunk) for chunk in tokens])} tokens")
    return tokens


def chatTokenLoader(repeat=True):
    while True:
        loader = chatTextLoader()
        for chunk in loader:
            lines = chunk.split("\n\n")
            tokens = []
            for line in lines:
                if line.startswith("User: "):
                    tokens.extend([userToken] + tokenize(line[6:]))
                elif line.startswith("+query"):
                    tokens.extend([queryToken] + tokenize(line[7:]))
                elif line.startswith("+get sources"):
                    tokens.extend([getSourcesToken] + tokenize(line[13:]))
                elif line.startswith("+read source"):
                    tokens.extend([readSourceToken] + tokenize(line[13:]))
                elif line.startswith("+respond"):
                    tokens.extend([respondToken] + tokenize(line[9:]))
                else:
                    continue
                tokens.extend(tokenize("\n\n"))
            tokens = tokens[:-2]

            yield np.array(tokens + [stopToken], dtype=np.uint8)

        if not repeat:
            break


def getAugmentedCriticData(tokens):
    newLineToken = stoi["\n"]

    randomize = False
    for i, token in enumerate(tokens):
        if token == newLineToken:
            randomize = False
        elif randomize:
            tokens[i] = np.random.randint(6, vocabSize, 1)
        elif token == respondToken:
            randomize = True

    return tokens
