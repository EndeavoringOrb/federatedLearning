import numpy as np

numbers = input("Enter weight numbers: ")
numbers = [int(item) for item in numbers.split(" ")]

weights = []
for num in numbers:
    weight = np.load(f"weights/{num}.npy")
    weights.append(weight)

found_unequal_weights = False
for i, weight in enumerate(weights):
    for j, otherWeight in enumerate(weights):
        if not np.all(weight == otherWeight):
            print(f"Weight {i} != Weight {j}")
            found_unequal_weights = True
if not found_unequal_weights:
    print(f"All weights are equal!")
