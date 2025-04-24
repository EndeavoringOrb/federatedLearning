import json
from utilities import tools
from utilities.model import *
import random
import misc_python_tools as mpt
import difflib
import Levenshtein
import os
import math
import heapq


class Tokenizer:
    def __init__(self):
        print(f"Initializing tokenizer...")
        self.vocab = "abcdefghijklmnopqrstuvwxyz"
        self.vocab += "0123456789"
        self.vocab += "()+-*/."
        self.vocab_set = set(self.vocab)
        self.vocab = sorted(list(self.vocab_set))

        self.tool_token = 0
        self.result_token = 1
        self.answer_token = 2
        self.capitalize_token = 3
        self.user_token = 4
        self.bot_token = 5

        # special tokens
        self.special_tokens = {
            self.tool_token: "\x00",  # tool token
            self.result_token: "\x01",  # result token
            self.answer_token: "\x02",  # answer token
            self.capitalize_token: "\x03",  # capitalize token
            self.user_token: "\x04",  # user token
            self.bot_token: "\x05",  # bot token
        }
        """
        user: <user_token>what is 2+2?<user_token>
        bot: <bot_token><tool_token>calculate<tool_token>2+2<tool_token><result_token>4<result_token><answer_token>4<answer_token><bot_token>
        """

        self.stoi = {char: num + 6 for num, char in enumerate(self.vocab)}
        self.itos = {v: k for k, v in self.stoi.items()}

        for k, v in self.special_tokens.items():
            self.stoi[v] = k
            self.itos[k] = v

        self.vocab_size = len(self.vocab) + len(self.special_tokens)
        self.valid_bot_tokens = [self.stoi[item] for item in self.vocab] + [
            self.tool_token,
            self.answer_token,
            self.capitalize_token,
        ]
        self.basic_vocab_tokens = [self.stoi[item] for item in self.vocab]

    def get_valid_bot_tokens(self):
        return self.valid_bot_tokens

    def get_basic_vocab_tokens(self):
        return self.basic_vocab_tokens

    def get_vocab_size(self):
        return self.vocab_size

    def decode(self, tokens: list[int]):
        if isinstance(tokens, int):
            tokens = [tokens]
        text = ""
        for tok in tokens:
            try:
                if tok == self.capitalize_token:
                    text += self.itos[tok].upper()
                else:
                    text += self.itos[tok]
            except KeyError:
                continue
        return text

    def pretty_decode(self, tokens: list[int]):
        if isinstance(tokens, int):
            tokens = [tokens]
        text = ""
        for tok in tokens:
            try:
                if tok == self.tool_token:
                    text += "<tool_token>"
                elif tok == self.result_token:
                    text += "<result_token>"
                elif tok == self.answer_token:
                    text += "<answer_token>"
                elif tok == self.user_token:
                    text += "<user_token>"
                elif tok == self.bot_token:
                    text += "<bot_token>"
                elif tok == self.capitalize_token:
                    text += self.itos[tok].upper()
                else:
                    text += self.itos[tok]
            except KeyError:
                continue
        return text

    def encode(self, text: str):
        tokens = []
        for char in text:
            try:
                lower = char.lower()
                if lower != char:
                    tokens.append(self.capitalize_token)
                tokens.append(self.stoi[lower])
            except KeyError:
                continue
        return tokens

    def prepareData(self, data: dict, add_role_token=False):
        tokens = []
        tokenInfo = []
        for item in data:
            if add_role_token:
                if item["role"] == "user":
                    role_token = self.user_token
                else:
                    role_token = self.bot_token
                item_tokens = [role_token] + self.encode(item["text"]) + [role_token]
            else:
                item_tokens = self.encode(item["text"])
            tokens.extend(item_tokens)
            tokenInfo.append({"role": item["role"], "length": len(item_tokens)})
        return tokens, tokenInfo


class ToolManager:
    def __init__(self, tokenizer: Tokenizer):
        self.tool_map = {"calculate": tools.calculate}
        self.select_constraints = self.initSelectConstraints(tokenizer)

    def initSelectConstraints(self, tokenizer: Tokenizer):
        keys = self.tool_map.keys()
        maxLen = max([len(key) for key in keys])
        constraints = []
        for i in range(maxLen):
            idxConstraints = [
                tokenizer.encode(key[i])[0] for key in keys if len(key) > i
            ]
            constraints.append(idxConstraints)
        return constraints

    def getSelectConstraints(self, idx):
        if idx >= 0 and idx < len(self.select_constraints):
            return self.select_constraints[idx]
        return []

    def process(self, select: str, command: str):
        """
        Call tools
        """
        if select in self.tool_map:
            try:
                return self.tool_map[select](command)
            except Exception as e:
                pass
        return ""


def getReward(output: str, correctAnswer: str):
    """
    Computes similarity between two strings in the range [0,1].
    It combines Levenshtein distance and sequence matching.

    Args:
        output (str): First input string.
        correctAnswer (str): Second input string.

    Returns:
        float: Similarity score between 0 and 1.
    """
    # Levenshtein similarity (1 - normalized edit distance)
    lev_sim = 1 - (
        Levenshtein.distance(output, correctAnswer)
        / max(len(output), len(correctAnswer))
    )

    # Sequence matcher ratio (accounts for longest contiguous matching blocks)
    seq_matcher_sim = difflib.SequenceMatcher(None, output, correctAnswer).ratio()

    # Final score as a weighted average
    return (lev_sim + seq_matcher_sim) / 2


def softmax(x):
    np.subtract(x, np.max(x), x)  # Subtract max for numerical stability
    np.exp(x, x)
    invSum = 1.0 / np.sum(x)
    np.multiply(x, invSum, x)


def log_softmax(x, axis=-1):
    np.subtract(x, np.max(x), x)  # Subtract max for numerical stability
    logsumexp = np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
    np.subtract(x, logsumexp, x)


def sample(logits, vocabSize) -> int:
    softmax(logits)  # logits -> probs
    token = np.random.choice(vocabSize, 1, p=logits)[0]
    return token.item()


def sampleConstrained(logits, validTokens, greedy=False) -> int:
    validLogits = logits[validTokens]
    if greedy:
        tokenIdx = np.argmax(validLogits)
    else:
        softmax(validLogits)  # logits -> probs
        tokenIdx = np.random.choice(len(validTokens), 1, p=validLogits)[0]
    token = validTokens[tokenIdx]
    return token


def getSeqProb(model: ChatModel, config, state: np.ndarray, token_seq):
    total_prob = 1
    state_copy = state.copy()
    for idx, tok in enumerate(token_seq):
        logits = model.getPred(
            weights,
            state_copy,
            config["hiddenSize"],
            config["vocabSize"],
            config["nLayers"],
        )
        softmax(logits)  # logits -> probs
        prob = logits[tok]
        total_prob *= prob
        if idx != (len(token_seq) - 1):
            state_copy = model.getNextState(
                weights,
                state_copy,
                tok,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )
    return total_prob.item()


def getSeqLogProb(model: ChatModel, config, state: np.ndarray, token_seq):
    total_log_prob = 0
    state_copy = state.copy()
    for idx, tok in enumerate(token_seq):
        logits = model.getPred(
            weights,
            state_copy,
            config["hiddenSize"],
            config["vocabSize"],
            config["nLayers"],
        )
        log_softmax(logits)
        total_log_prob += logits[tok]
        if idx != (len(token_seq) - 1):
            state_copy = model.getNextState(
                weights,
                state_copy,
                tok,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )
    return total_log_prob.item()


def evaluateOLD(
    model: ChatModel,
    weights,
    tokenizer: Tokenizer,
    tokens,
    tokenInfo,
    config,
    toolManager: ToolManager,
):
    """
    tokenInfo:
        [{
            "role": str ("user" or "bot"),
            "length": int
        }]
    """
    # this is for batchSize=1
    state = model.getInitState(weights, config["hiddenSize"])
    tokensIdx = 0
    total_reward = 0
    generated_tokens = []

    for itemIdx, item in enumerate(tokenInfo):
        itemTokens = tokens[tokensIdx : tokensIdx + item["length"]]
        in_tool_select = False
        tool_select = ""
        in_tool_call = False
        tool_call = ""

        if item["role"] == "user":
            itemTokens = (
                [tokenizer.user_token] + itemTokens + [tokenizer.user_token]
            )  # add user token on both ends

            for tok in itemTokens:
                state = model.getNextState(
                    weights,
                    state,
                    tok,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )
        elif item["role"] == "bot":
            # process initial bot token
            generated_tokens.append(tokenizer.bot_token)
            state = model.getNextState(
                weights,
                state,
                tokenizer.bot_token,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )

            for i in range(config["maxNewTokens"]):
                logits = model.getPred(
                    weights,
                    state,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )

                # get valid tokens
                valid_tokens = None
                if in_tool_select:
                    constraints = toolManager.getSelectConstraints(len(tool_select))
                    if len(constraints) > 0:
                        valid_tokens = constraints
                    else:
                        valid_tokens = [tokenizer.tool_token]
                elif in_tool_call:
                    valid_tokens = tokenizer.get_basic_vocab_tokens() + [
                        tokenizer.tool_token,
                        tokenizer.capitalize_token,
                    ]

                # sample token
                if valid_tokens is None:
                    valid_tokens = tokenizer.get_valid_bot_tokens()
                tok = sampleConstrained(logits, valid_tokens, greedy=True)

                # update state
                toks_to_process = [tok]
                if in_tool_select and tok not in tokenizer.special_tokens:
                    tool_select += tokenizer.decode(tok)
                if in_tool_call and tok not in tokenizer.special_tokens:
                    tool_call += tokenizer.decode(tok)
                if tok == tokenizer.tool_token:
                    if not in_tool_call and not in_tool_select:  # start tool call
                        in_tool_select = True
                    elif in_tool_select:  # finished tool select
                        in_tool_select = False
                        in_tool_call = True
                    elif in_tool_call:  # finished tool call
                        in_tool_call = False
                        result = toolManager.process(tool_select, tool_call)
                        result_tokens = (
                            [tokenizer.result_token]
                            + tokenizer.encode(result)
                            + [tokenizer.result_token]
                        )
                        toks_to_process.extend(result_tokens)
                        tool_select = ""
                        tool_call = ""
                elif tok == tokenizer.answer_token:
                    break

                generated_tokens.extend(toks_to_process)

                for tok in toks_to_process:
                    state = model.getNextState(
                        weights,
                        state,
                        tok,
                        config["hiddenSize"],
                        config["vocabSize"],
                        config["nLayers"],
                    )

            toks_to_process = []
            target_seq = (
                [tokenizer.answer_token] + itemTokens + [tokenizer.answer_token]
            )
            if in_tool_select:
                result = toolManager.process(tool_select, tool_call)
                result_tokens = (
                    [tokenizer.result_token]
                    + tokenizer.encode(result)
                    + [tokenizer.result_token]
                )
                toks_to_process = 2 * [tokenizer.tool_token] + result_tokens
            elif in_tool_call:
                result = toolManager.process(tool_select, tool_call)
                result_tokens = (
                    [tokenizer.result_token]
                    + tokenizer.encode(result)
                    + [tokenizer.result_token]
                )
                toks_to_process = [tokenizer.tool_token] + result_tokens

            reward = getSeqProb(
                model, config, state, target_seq
            )  # 0-1, probability that the model would have outputted the correct answer
            total_reward += reward
            generated_tokens.extend(target_seq)

            if itemIdx != (
                len(tokenInfo) - 1
            ):  # if this is the last item, we don't need to process these tokens
                # process reward tokens
                for tok in toks_to_process + target_seq:
                    state = model.getNextState(
                        weights,
                        state,
                        tok,
                        config["hiddenSize"],
                        config["vocabSize"],
                        config["nLayers"],
                    )

                # process final bot token
                state = model.getNextState(
                    weights,
                    state,
                    tokenizer.bot_token,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )

            generated_tokens.append(tokenizer.bot_token)

        tokensIdx += item["length"]

    num_responses = [item["role"] for item in tokenInfo].count("bot")
    if num_responses == 0:  # avoid div by zero error
        return 0
    return total_reward / num_responses  # normalize reward by number of responses


class TextState:
    def __init__(self):
        self.in_tool_select = False
        self.tool_select = ""
        self.in_tool_call = False
        self.tool_call = ""
        self.is_first_tok = True

    def copy(self, other):
        self.in_tool_select = other.in_tool_select
        self.tool_select = other.tool_select
        self.in_tool_call = other.in_tool_call
        self.tool_call = other.tool_call
        self.is_first_tok = other.is_first_tok

    def get_valid_tokens(self, tokenizer: Tokenizer):
        # get valid tokens
        if self.is_first_tok:
            return [tokenizer.bot_token]

        valid_tokens = None
        if self.in_tool_select:
            constraints = toolManager.getSelectConstraints(len(self.tool_select))
            if len(constraints) > 0:
                valid_tokens = constraints
            else:
                valid_tokens = [tokenizer.tool_token]
        elif self.in_tool_call:
            valid_tokens = tokenizer.get_basic_vocab_tokens() + [
                tokenizer.tool_token,
                tokenizer.capitalize_token,
            ]

        if valid_tokens is None:
            valid_tokens = tokenizer.get_valid_bot_tokens()
        return valid_tokens

    def update(self, tok, tokenizer: Tokenizer, toolManager: ToolManager):
        # update state
        toks_to_process = [tok]
        self.is_first_tok = False
        if self.in_tool_select and tok not in tokenizer.special_tokens:
            self.tool_select += tokenizer.decode(tok)
        if self.in_tool_call and tok not in tokenizer.special_tokens:
            self.tool_call += tokenizer.decode(tok)
        if tok == tokenizer.tool_token:
            if not self.in_tool_call and not self.in_tool_select:  # start tool call
                self.in_tool_select = True
            elif self.in_tool_select:  # finished tool select
                self.in_tool_select = False
                self.in_tool_call = True
            elif self.in_tool_call:  # finished tool call
                self.in_tool_call = False
                result = toolManager.process(self.tool_select, self.tool_call)
                result_tokens = (
                    [tokenizer.result_token]
                    + tokenizer.encode(result)
                    + [tokenizer.result_token]
                )
                toks_to_process.extend(result_tokens)
                self.tool_select = ""
                self.tool_call = ""
        return toks_to_process


class Node:
    def __init__(self, state, log_prob, text_state: TextState, depth, seq):
        self.state = state
        self.log_prob = log_prob
        self.text_state = text_state
        self.children = []
        self.depth = depth
        self.seq = seq


def beam_search_top_sequences(
    model: ChatModel,
    weights: np.ndarray,
    init_state: np.ndarray,
    config,
    tokenizer: Tokenizer,
    toolManager: ToolManager,
    itemTokens: list[int],
):
    top_n = config["top_n_seqs"]
    max_len = config.get("max_depth", 20)  # optional cap on depth

    target_seq = [tokenizer.answer_token] + itemTokens + [tokenizer.answer_token]

    root = Node(init_state, 0.0, TextState(), 0, [])
    beam = [root]
    final_heap = []

    for _ in range(max_len):
        candidates = []
        for node in beam:
            valid_tokens = node.text_state.get_valid_tokens(tokenizer)
            logits = model.getPred(
                weights,
                node.state,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )
            log_softmax(logits)

            for tok in valid_tokens:
                if tok == tokenizer.answer_token:
                    # Try completing the sequence and evaluate
                    logp = getSeqLogProb(model, config, node.state, target_seq)
                    total_log_prob = node.log_prob + logp
                    final_heap.append((total_log_prob.item(), node.state))
                    final_heap.sort(key=lambda x: x[0])
                    if len(final_heap) > top_n:
                        final_heap.pop(0)
                    continue

                new_log_prob = node.log_prob + logits[tok]

                # update text state
                new_text_state = TextState()
                new_text_state.copy(node.text_state)
                toks_to_process = new_text_state.update(tok, tokenizer, toolManager)

                # update model state
                new_state = node.state.copy()
                for process_tok in toks_to_process:
                    new_state = model.getNextState(
                        weights,
                        new_state,
                        process_tok,
                        config["hiddenSize"],
                        config["vocabSize"],
                        config["nLayers"],
                    )

                new_node = Node(
                    new_state,
                    new_log_prob,
                    new_text_state,
                    node.depth + 1,
                    node.seq + toks_to_process,
                )
                candidates.append(new_node)

        if not candidates:
            break

        # Keep top_n nodes only
        candidates.sort(key=lambda x: x.log_prob, reverse=True)
        beam = candidates[:top_n]

    return sorted(final_heap, reverse=True), target_seq  # return highest-prob first


def find_top_sequences(
    model: ChatModel,
    weights: np.ndarray,
    init_state: np.ndarray,
    config,
    tokenizer: Tokenizer,
    toolManager: ToolManager,
    itemTokens: list[int],
):
    root = Node(init_state, 0.0, TextState(), 0, [])
    max_heap = []
    max_log_prob = -math.inf
    target_seq = [tokenizer.answer_token] + itemTokens + [tokenizer.answer_token]
    top_n = config["top_n_seqs"]

    # init heap with beam search so that our initial max_log_prob is higher and we can cut off more nodes
    max_heap, _ = beam_search_top_sequences(
        model, weights, init_state, config, tokenizer, toolManager, itemTokens
    )
    max_log_prob = max_heap[-1][0]

    def postorder(node: Node):
        nonlocal max_log_prob

        # get valid tokens
        valid_tokens = node.text_state.get_valid_tokens(tokenizer)

        if tokenizer.answer_token in valid_tokens:
            # Check if the probability of this seq ending with the target_suffix is higher than any of the current top_n sequences
            logp = getSeqLogProb(model, config, node.state, target_seq)
            new_prob = node.log_prob + logp
            if new_prob > max_log_prob:
                if all([np.array_equal(node.state, item[1]) for item in max_heap]): # avoid duplicates from the beam search
                    max_heap.append((new_prob.item(), node.state))
                    max_heap.sort(key=lambda x: x[0])
                    if len(max_heap) > top_n:
                        max_heap.pop(0)
                        max_log_prob = max_heap[0][0]
            else:
                return # this sequence should be cut off

        logits = model.getPred(
            weights,
            node.state,
            config["hiddenSize"],
            config["vocabSize"],
            config["nLayers"],
        )
        log_softmax(logits)

        for tok in valid_tokens:
            if tok == tokenizer.answer_token:
                continue

            log_prob = logits[tok]
            new_log_prob = node.log_prob + log_prob
            if new_log_prob < max_log_prob:
                continue  # prune this tok's branch

            # update text state
            new_text_state = TextState()
            new_text_state.copy(node.text_state)
            toks_to_process = new_text_state.update(tok, tokenizer, toolManager)

            # update model state
            new_state = node.state.copy()
            for process_tok in toks_to_process:
                new_state = model.getNextState(
                    weights,
                    new_state,
                    process_tok,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )

            # create child node
            child = Node(
                new_state,
                new_log_prob,
                new_text_state,
                node.depth + 1,
                node.seq + toks_to_process,
            )
            node.children.append(child)
            postorder(child)

            # check if the parent node should be pruned
            if node.log_prob < max_log_prob:
                return

    postorder(root)

    return sorted(max_heap, reverse=True), target_seq  # highest prob first


def evaluate(
    model: ChatModel,
    weights,
    tokenizer: Tokenizer,
    tokens,
    tokenInfo,
    config,
    toolManager: ToolManager,
):
    """
    tokenInfo:
        [{
            "role": str ("user" or "bot"),
            "length": int
        }]
    """
    # this is for batchSize=1
    state = model.getInitState(weights, config["hiddenSize"])
    tokensIdx = 0
    total_reward = 0
    top_n = config["top_n_seqs"]

    for itemIdx, item in enumerate(tokenInfo):
        itemTokens = tokens[tokensIdx : tokensIdx + item["length"]]
        tokensIdx += item["length"]

        if item["role"] == "user":
            itemTokens = (
                [tokenizer.user_token] + itemTokens + [tokenizer.user_token]
            )  # add user token on both ends

            for tok in itemTokens:
                state = model.getNextState(
                    weights,
                    state,
                    tok,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )
        elif item["role"] == "bot":
            top_seqs, target_seq = find_top_sequences(
                model, weights, state, config, tokenizer, toolManager, itemTokens
            )

            reward = sum([item[0] for item in top_seqs])  # sum of top_n seq log probs
            total_reward += reward / top_n

            state = top_seqs[0][1]  # set state to the highest prob state

            if itemIdx != (
                len(tokenInfo) - 1
            ):  # if this is the last item, we don't need to process these tokens because there is no next turn relying on this state
                # process target_seq tokens
                for tok in target_seq:
                    state = model.getNextState(
                        weights,
                        state,
                        tok,
                        config["hiddenSize"],
                        config["vocabSize"],
                        config["nLayers"],
                    )

                # process final bot token
                state = model.getNextState(
                    weights,
                    state,
                    tokenizer.bot_token,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )

    num_responses = [item["role"] for item in tokenInfo].count("bot")
    if num_responses == 0:  # avoid div by zero error
        return 0
    return total_reward / num_responses  # normalize reward by number of responses


def save_checkpoint(config, weights):
    """Save model checkpoint"""
    checkpoint_path = f"{config['save_folder']}/{config['run_num']}"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save config
    config_path = f"{checkpoint_path}/config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # Save model
    model_data = np.concatenate(
        (
            [
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            ],
            weights,
        )
    )
    np.save(f"{checkpoint_path}/model.npy", model_data)

    mpt.log_sys(f"Checkpoint saved.")


if __name__ == "__main__":
    mpt.log_sys("Initializing...")
    config = {
        "population": 64,
        "numTrials": 1,  # number of trials per population member
        "learningRate": 1e-3,
        "sigma": 1e-5,
        "hiddenSize": 16,
        "nLayers": 4,
        "optimizer": "adam",  # "sgd" or "adam"
        "beta1": 0.9,
        "beta2": 0.999,
        "stepNum": 0,
        "maxNewTokens": 100,
        "maximize": True,
        "use_current_reward_as_mean": False,
        "save_folder": "data/tool/trainingRuns",
        "top_n_seqs": 5,
    }
    os.makedirs(config["save_folder"], exist_ok=True)
    thing = os.listdir(config["save_folder"])
    thing = [item for item in thing if item.isdigit()]
    if len(thing) == 0:
        config["run_num"] = 0
    else:
        max_run_num = max(
            [int(num) for num in os.listdir(config["save_folder"]) if num.isdigit()]
        )
        config["run_num"] = max_run_num + 1
    config["best_reward"] = float("-inf") if config["maximize"] else float("inf")
    random.seed(42)
    np.random.seed(43)
    tokenizer = Tokenizer()
    config["vocabSize"] = tokenizer.get_vocab_size()
    model = ChatModel()
    toolManager = ToolManager(tokenizer)

    weights = model.getWeights(
        config["hiddenSize"], config["vocabSize"], config["nLayers"]
    )
    grad: np.ndarray = np.zeros_like(weights)
    nParams = weights.shape[0]
    print(f"Model has {nParams:,} parameters")
    optimizerValues = np.zeros(nParams * 2).astype(np.float32)
    newWeights = False
    all_rewards = np.zeros(config["population"])
    reward_info = []

    optimizer = AdamOptimizer(
        weights.shape[0],
        config["learningRate"],
        config["beta1"],
        config["beta2"],
    )
    optimizer.m = optimizerValues[: weights.shape[0]]
    optimizer.v = optimizerValues[weights.shape[0] :]

    # with open("data/toolData/unsupervised.json", "r", encoding="utf-8") as f:
    #     data: list[dict] = json.load(f)
    data = [
        [{"role": "user", "text": "What is 2+2?"}, {"role": "bot", "text": "4"}],
    ]

    tokenData = []
    for item in data:
        tokenData.append(tokenizer.prepareData(item))

    while True:
        # Update trackers
        config["stepNum"] += 1
        mpt.log(f"Step {config['stepNum']}")

        # Set random seed
        seed = random.randint(0, 1e6)
        np.random.seed(seed)

        # Run trials
        for i in range(config["population"]):
            trial_weights = (
                weights + np.random.randn(weights.shape[0]) * config["sigma"]
            )
            total_reward = 0
            if i > 0:
                mpt.log_clear(1)
            for j in range(config["numTrials"]):
                if j > 0:
                    mpt.log_clear(1)
                mpt.log(
                    f"Weight [{i+1}/{config['population']}], Trial [{j+1}/{config['numTrials']}]"
                )
                for tokens, tokenInfo in tokenData:
                    reward = evaluate(
                        model,
                        trial_weights,
                        tokenizer,
                        tokens,
                        tokenInfo,
                        config,
                        toolManager,
                    )
                    total_reward += reward
            total_reward /= config["numTrials"] * len(tokenData)
            all_rewards[i] = reward

        # Run trials on current weights
        current_reward = 0
        for j in range(config["numTrials"]):
            for tokens, tokenInfo in tokenData:
                reward = evaluate(
                    model,
                    weights,
                    tokenizer,
                    tokens,
                    tokenInfo,
                    config,
                    toolManager,
                )
                current_reward += reward
        current_reward /= config["numTrials"] * len(tokenData)

        if config["use_current_reward_as_mean"]:
            mean = current_reward
        else:
            mean = all_rewards.mean().item()

        # Normalize rewards
        std = all_rewards.std().item()
        max_reward = all_rewards.max().item()
        if np.isnan(mean) or np.isnan(std) or std == 0.0:
            mpt.log(f"Invalid rewards")
            continue
        else:
            mpt.log(f"Mean reward: {mean}")
            mulVal = 1.0 / (
                np.std(all_rewards) * float(config["population"]) * config["sigma"]
            )
            all_rewards = (all_rewards - mean) * mulVal

        # Update weights
        mpt.log(f"Updating weights")
        grad.fill(0)
        np.random.seed(seed)
        for trial in range(config["population"]):
            grad += np.random.randn(weights.shape[0]) * all_rewards[trial]

        if config["optimizer"] == "adam":
            grad = optimizer.getGrad(grad)
        else:
            grad *= config["learningRate"]

        if config["maximize"]:
            weights += grad
        else:
            weights -= grad

        mpt.log_clear(4)
        mpt.log(f"Step {config['stepNum']}, Avg: {mean}, Max: {max_reward}")

        # Save loss.csv by appending
        checkpoint_path = f"{config['save_folder']}/{config['run_num']}"
        os.makedirs(checkpoint_path, exist_ok=True)
        loss_path = f"{checkpoint_path}/loss.csv"
        write_header = not os.path.exists(loss_path)

        with open(loss_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("current_reward,mean,max_reward\n")
            f.write(",".join(map(str, (current_reward, mean, max_reward))) + "\n")

        if config["maximize"]:
            if current_reward > config["best_reward"]:
                config["best_reward"] = current_reward
                save_checkpoint(config, weights)
        else:
            if current_reward < config["best_reward"]:
                config["best_reward"] = current_reward
                save_checkpoint(config, weights)
