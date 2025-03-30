import re
from utilities import tools
from utilities.model import *
import random
import misc_python_tools as mpt
import difflib
import Levenshtein


class Tokenizer:
    def __init__(self):
        print(f"Initializing tokenizer...")
        self.vocab = "abcdefghijklmnopqrstuvwxyz"
        self.vocab += "0123456789"
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

        self.stoi = {char: num + 6 for num, char in enumerate(self.vocab)}
        self.itos = {v: k for k, v in self.stoi.items()}

        for k, v in self.special_tokens.items():
            self.stoi[v] = k
            self.itos[k] = v

    def decode(self, tokens: list[int]):
        if isinstance(tokens, int):
            tokens = [tokens]
        text = ""
        for tok in tokens:
            try:
                text += self.itos[tok]
            except KeyError:
                continue
        return text

    def encode(self, text: str):
        tokens = []
        for char in text:
            try:
                tokens.append(self.stoi[char])
            except KeyError:
                continue
        return tokens

    def prepareData(self, data: dict):
        tokens = []
        tokenInfo = []
        for item in data:
            if item["role"] == "user":
                role_token = self.user_token
            else:
                role_token = self.bot_token
            item_tokens = [role_token] + self.encode(item["text"]) + [role_token]
            tokens.extend(item_tokens)
            tokenInfo.append({"role": item["role"], "length": len(item_tokens)})
        return tokens, tokenInfo


def toolProcess(text: str):
    """
    Replace any function calls with their output
    i.e.
    "The answer is calculate(2+2)" -> "The answer is 4"
    """
    tool_map = {"calculate": tools.calculate}
    for name, func in tool_map.items():
        pattern = rf"{name}\((.*?)\)"
        matches = re.findall(pattern, text)
        if len(matches) == 1:
            return func(text)
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


def sample(logits, vocabSize):
    softmax(logits)  # logits -> probs
    token = np.random.choice(vocabSize, 1, p=logits)[0]
    return token.item()


def evaluate(
    model: ChatModel, weights, tokenizer: Tokenizer, tokens, tokenInfo, config
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
    generated_tokens = []
    in_tool_call = False
    tool_call = ""
    in_answer = False
    answer = ""
    total_reward = 0

    for item in tokenInfo:
        itemTokens = tokens[tokensIdx : tokensIdx + item["length"]]

        if item["role"] == "user":
            for idx in range(tokensIdx, tokensIdx + item["length"]):
                state = model.getNextState(
                    weights,
                    state,
                    tokens[idx],
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )
        elif item["role"] == "bot":
            for idx in range(config["maxNewTokens"]):
                logits = model.getPred(
                    weights,
                    state,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )
                tok = sample(logits, config["vocabSize"])
                generated_tokens.append(tok)
                if tok == tokenizer.answer_token:
                    if in_answer:
                        break
                    in_answer = True
                if tok == tokenizer.tool_token:
                    if in_tool_call:
                        result = toolProcess(tool_call)
                        result_tokens = (
                            [tokenizer.result_token]
                            + tokenizer.encode(result)
                            + [tokenizer.result_token]
                        )
                        for tool_call_result_tok in result_tokens:
                            state = model.getNextState(
                                weights,
                                state,
                                tool_call_result_tok,
                                config["hiddenSize"],
                                config["vocabSize"],
                                config["nLayers"],
                            )
                        tool_call = ""
                    in_tool_call = not in_tool_call
                if in_tool_call and tok not in tokenizer.special_tokens:
                    tool_call += tokenizer.decode(tok)
                if in_answer and tok not in tokenizer.special_tokens:
                    answer += tokenizer.decode(tok)

            reward = getReward(answer, tokenizer.decode(itemTokens))
            total_reward += reward

        tokensIdx += item["length"]

    return total_reward


if __name__ == "__main__":
    mpt.log_sys("Initializing...")
    config = {
        "numTrials": 1000,
        "learningRate": 1e-3,
        "sigma": 1e-2,
        "hiddenSize": 16,
        "vocabSize": 76,
        "nLayers": 4,
        "optimizer": "adam",  # "sgd" or "adam"
        "beta1": 0.9,
        "beta2": 0.999,
        "stepNum": 0,
        "maxNewTokens": 100,
    }
    tokenizer = Tokenizer()
    model = ChatModel()

    weights = model.getWeights(
        config["hiddenSize"], config["vocabSize"], config["nLayers"]
    )
    grad: np.ndarray = np.zeros_like(weights)
    nParams = weights.shape[0]
    print(f"Model has {nParams:,} parameters")
    optimizerValues = np.zeros(nParams * 2).astype(np.float32)
    newWeights = False
    all_rewards = np.zeros(config["numTrials"])
    reward_info = []
    stepNum = 0

    optimizer = AdamOptimizer(
        weights.shape[0],
        config["learningRate"],
        config["beta1"],
        config["beta2"],
    )
    optimizer.m = optimizerValues[: weights.shape[0]]
    optimizer.v = optimizerValues[weights.shape[0] :]

    data = [{"role": "user", "text": "What is 2+2?"}, {"role": "bot", "text": "4"}]

    tokens, tokenInfo = tokenizer.prepareData(data)

    while True:
        # Update trackers
        stepNum += 1
        mpt.log(f"Step {stepNum}")

        # Set random seed
        seed = random.randint(0, 1e6)
        np.random.seed(seed)

        # Run trials
        mpt.log(f"Running {config['numTrials']} trials")
        for i in range(config["numTrials"]):
            if i > 0:
                mpt.log_clear(1)
            mpt.log(f"  {i+1}/{config['numTrials']}")
            trial_weights = (
                weights + np.random.randn(weights.shape[0]) * config["sigma"]
            )
            reward = evaluate(
                model, trial_weights, tokenizer, tokens, tokenInfo, config
            )
            all_rewards[i] = reward
        if config["numTrials"] > 0:
            mpt.log_clear(1)

        # Normalize rewards
        mean = all_rewards.mean()
        std = all_rewards.std()
        if np.isnan(mean) or np.isnan(std) or std == 0.0:
            mpt.log(f"Invalid rewards")
            continue
        else:
            mpt.log(f"Mean reward: {mean.item()}")
            mulVal = 1.0 / (
                np.std(all_rewards) * float(config["numTrials"]) * config["sigma"]
            )
            all_rewards = (all_rewards - mean) * mulVal

        # Update weights
        mpt.log(f"Updating weights")
        grad.fill(0)
        np.random.seed(seed)
        for trial in range(config["numTrials"]):
            grad += np.random.randn(weights.shape[0]) * all_rewards[trial]

        if config["optimizer"] == "adam":
            grad = optimizer.getGrad(grad)
        else:
            grad *= config["learningRate"]

        weights -= grad
