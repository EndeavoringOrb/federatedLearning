import json
from collections import Counter
from shakespeareData import textLoader


def getVocab():
    vocab = Counter()
    numTrainingExamples = 0

    for chunk in textLoader():
        vocab.update(chunk)
        numTrainingExamples += 1

    print(f"There are {numTrainingExamples} training examples.")
    # Sort the vocabulary by frequency in descending order
    return vocab.most_common()

if __name__ == "__main__":
    vocab = getVocab()

    # Print sorted vocab and its size
    for character, frequency in vocab:
        print(f"{character}: {frequency}")
    print(f"Vocab Size: {len(vocab)}")

    with open("shakespeareFreqs.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)