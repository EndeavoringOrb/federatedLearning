import heapq
import math


# Dummy model: returns log probabilities (replace with your own model logic)
def token_log_prob(seq, tok):
    # For example purposes: assign random or heuristic-based log-probs
    import random

    return -random.random() * 2  # log prob in range [-2, 0]


# Parameters
vocab = ["the", "cat", "sat", "on", "mat", "end", "</s>"]
max_len = 6
top_n = 10
target_suffix = ["</s>"]
valid_vocab = [item for item in vocab if item not in target_suffix]


# Node representation
class Node:
    def __init__(self, seq, log_prob):
        self.seq = seq
        self.log_prob = log_prob
        self.children = []


def ends_with(seq, suffix):
    return seq[-len(suffix) :] == suffix if len(seq) >= len(suffix) else False


def get_seq_log_prob(seq, target_seq):
    log_prob = 0
    for i in range(len(target_seq)):
        log_prob += token_log_prob(seq + target_seq[:i], target_seq[i])
    return log_prob


# Search function
def find_top_sequences():
    root = Node([], 0.0)
    max_heap = []
    max_log_prob = -math.inf

    def postorder(node: Node):
        nonlocal max_log_prob

        # Check if the probability of this seq ending with the target_suffix is higher than any of the current top_n sequences
        logp = get_seq_log_prob(node.seq, target_suffix)
        if node.log_prob + logp > max_log_prob:
            heapq.heappush(
                max_heap, (node.log_prob + logp, list(node.seq + target_suffix))
            )
            if len(max_heap) > top_n:
                heapq.heappop(max_heap)
                max_log_prob = max_heap[0][0]

        for tok in valid_vocab:
            new_seq = node.seq + [tok]
            logp = token_log_prob(node.seq, tok)
            new_log_prob = node.log_prob + logp
            if new_log_prob < max_log_prob:
                continue  # prune this branch
            child = Node(new_seq, new_log_prob)
            node.children.append(child)
            postorder(child)

    postorder(root)

    return sorted(max_heap, reverse=True)  # highest prob first


# Run the search
top_sequences = find_top_sequences()
for log_prob, seq in top_sequences:
    print(f"Log Prob: {log_prob:.4f}, Sequence: {' '.join(seq)}")
