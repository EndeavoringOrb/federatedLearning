from shakespeareData import *

textChunks = [chunk for chunk in textLoader()]

while True:
    query = input("Enter phrase to search for: ")
    for i, chunk in enumerate(textChunks):
        if query in chunk:
            print(f"Chunk {i:,}:")
            for character in chunk:
                print(character, end="")
            print("\n", flush=True)
    print()