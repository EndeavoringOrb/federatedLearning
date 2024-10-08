import os
import json
from tqdm import tqdm
from collections import Counter


def getVocab():
    vocab = Counter()
    filenames = os.listdir("redditData")
    numLines = 0

    for filename in tqdm(filenames, desc="Getting vocab"):
        fileVocab = Counter()
        # Open and read the file
        with open(f"redditData/{filename}", "r") as file:
            for line in file:
                # Parse each line
                item = json.loads(line)

                # Update vocab with characters from the title and body
                fileVocab.update(item["title"])
                fileVocab.update(item["body"])

                # Update vocab with characters from each answer's body
                for answer in item["answers"]:
                    fileVocab.update(answer["body"])
                    numLines += 1

                if len(fileVocab) > 500:
                    vocab.update(fileVocab)
                    fileVocab = Counter()

    vocab.update(fileVocab)

    print(f"There are {numLines} training examples.")
    # Sort the vocabulary by frequency in descending order
    return vocab.most_common()

if __name__ == "__main__":
    vocab = getVocab()

    # Print sorted vocab and its size
    for character, frequency in vocab:
        print(f"{character}: {frequency}")
    print(f"Vocab Size: {len(vocab)}")

    with open("redditFreqs.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)