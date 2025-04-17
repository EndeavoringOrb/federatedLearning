import json
import numpy as np
import os
import argparse
from toolTrain import ChatModel, Tokenizer, ToolManager, sampleConstrained


def load_checkpoint(checkpoint_dir):
    """Load model checkpoint from directory"""
    # Load config
    config_path = f"{checkpoint_dir}/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load model weights
    model_data = np.load(f"{checkpoint_dir}/model.npy")

    # First 3 values are hiddenSize, vocabSize, nLayers
    config["hiddenSize"] = int(model_data[0])
    config["vocabSize"] = int(model_data[1])
    config["nLayers"] = int(model_data[2])

    # Rest is weights
    weights = model_data[3:]

    return config, weights


def generate_response(
    model: ChatModel,
    weights: np.ndarray,
    state: np.ndarray,
    tokenizer: Tokenizer,
    config: dict,
    toolManager: ToolManager,
    max_tokens=100,
):
    """Generate a response from the model"""
    generated_tokens = [tokenizer.bot_token]
    num_tokens_processed = 0

    # Process initial bot token for state
    state = model.getNextState(
        weights,
        state,
        tokenizer.bot_token,
        config["hiddenSize"],
        config["vocabSize"],
        config["nLayers"],
    )

    # Track tool state
    in_tool_select = False
    tool_select = ""
    in_tool_call = False
    tool_call = ""
    in_answer = False

    # Generate tokens
    for _ in range(max_tokens):
        logits = model.getPred(
            weights,
            state,
            config["hiddenSize"],
            config["vocabSize"],
            config["nLayers"],
        )

        # Get valid tokens based on context
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

        # Sample token
        if valid_tokens is None:
            valid_tokens = tokenizer.get_valid_bot_tokens()
        tok = sampleConstrained(logits, valid_tokens, greedy=False)

        # Update state and handle special tokens
        toks_to_process = [tok]
        if in_tool_select and tok not in tokenizer.special_tokens:
            tool_select += tokenizer.decode(tok)
        if in_tool_call and tok not in tokenizer.special_tokens:
            tool_call += tokenizer.decode(tok)

        if tok == tokenizer.tool_token:
            if not in_tool_call and not in_tool_select:  # Start tool selection
                in_tool_select = True
            elif in_tool_select:  # Finished tool selection
                in_tool_select = False
                in_tool_call = True
            elif in_tool_call:  # Finished tool call
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
            if not in_answer:
                in_answer = True
            else:
                in_answer = False
                generated_tokens.append(tok)
                break
        elif tok == tokenizer.bot_token:
            # End of bot response
            generated_tokens.append(tok)
            break

        generated_tokens.extend(toks_to_process)

        for tok in toks_to_process:
            num_tokens_processed += 1
            state = model.getNextState(
                weights,
                state,
                tok,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )

    # Process any remaining tool operations
    if in_tool_select or in_tool_call:
        result = toolManager.process(tool_select, tool_call)
        result_tokens = (
            [tokenizer.result_token]
            + tokenizer.encode(result)
            + [tokenizer.result_token]
        )

        if in_tool_select:
            toks_to_process = 2 * [tokenizer.tool_token] + result_tokens
        else:  # in_tool_call
            toks_to_process = [tokenizer.tool_token] + result_tokens

        generated_tokens.extend(toks_to_process)
    num_answer_tokens = generated_tokens.count(tokenizer.answer_token)
    if in_answer:
        generated_tokens.append(tokenizer.answer_token)
    elif num_answer_tokens == 1:
        generated_tokens.append(tokenizer.answer_token)
    elif num_answer_tokens == 0:
        generated_tokens.extend([tokenizer.answer_token] * 2)

    # Add final bot token if not already present
    if generated_tokens[-1] != tokenizer.bot_token:
        generated_tokens.append(tokenizer.bot_token)

    for tok in generated_tokens[num_tokens_processed:]:
        state = model.getNextState(
            weights,
            state,
            tok,
            config["hiddenSize"],
            config["vocabSize"],
            config["nLayers"],
        )

    return generated_tokens, state


def format_bot_response(tokens, tokenizer):
    """Format bot response for display to user"""
    text = tokenizer.pretty_decode(tokens)

    # Clean up special tokens for nicer display
    text = text.replace("<bot_token>", "")
    text = text.replace("<tool_token>", "[TOOL]")
    text = text.replace("<result_token>", "[RESULT]")
    text = text.replace("<answer_token>", "[ANSWER]")

    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Test a trained tool-using language model"
    )
    parser.add_argument("--run_num", type=int, help="Run number to load model from")
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Directory containing model checkpoint"
    )
    parser.add_argument(
        "--max_turns", type=int, default=10, help="Maximum conversation turns"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum tokens per response"
    )
    args = parser.parse_args()

    # Set up model path
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        save_folder = "data/tool/trainingRuns"
        run_num = args.run_num
        if run_num is None:
            # Find latest run if none specified
            runs = [int(d) for d in os.listdir(save_folder) if d.isdigit()]
            if not runs:
                print("No trained models found. Please train a model first.")
                return
            run_num = max(runs)
            print(f"Using latest run: {run_num}")
        checkpoint_dir = f"{save_folder}/{run_num}"

    # Load model
    print(f"Loading model from {checkpoint_dir}...")
    config, weights = load_checkpoint(checkpoint_dir)

    # Initialize components
    tokenizer = Tokenizer()
    model = ChatModel()
    toolManager = ToolManager(tokenizer)

    # Init model state
    state = model.getInitState(weights, config["hiddenSize"])

    print("\n--- Conversation with Tool-Using AI Model ---")
    print("Type 'quit' or 'exit' to end the conversation.")

    for turn in range(args.max_turns):
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Update state
        item_tokens = (
            [tokenizer.user_token]
            + tokenizer.encode(user_input)
            + [tokenizer.user_token]
        )
        for tok in item_tokens:
            state = model.getNextState(
                weights,
                state,
                tok,
                config["hiddenSize"],
                config["vocabSize"],
                config["nLayers"],
            )

        # Generate response
        response_tokens, state = generate_response(
            model,
            weights,
            state,
            tokenizer,
            config,
            toolManager,
            max_tokens=args.max_tokens,
        )

        # Display response
        formatted_response = format_bot_response(response_tokens, tokenizer)
        print(f"\nAI: {formatted_response}")

    print("\n--- End of Conversation ---")


if __name__ == "__main__":
    main()
