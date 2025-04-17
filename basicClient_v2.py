from utilities.communication import CommunicationHandler, log
from utilities.model import *
from time import perf_counter
import numpy as np


if __name__ == "__main__":
    # Server settings
    server_ip = "endeavoringorb.com"
    server_port = 55551
    log.info("Starting client...")

    # Initialize CommunicationHandler instance for the client
    comm = CommunicationHandler(server_ip, server_port)

    if not comm.connect():
        log.critical(f"Failed to connect to server {server_ip}:{server_port}. Exiting.")
        exit(0)

    try:
        # Init trackers
        stepStart = perf_counter()
        timePerTrial = 0  # Initialize estimate for first step

        # Receive initial data bundle (6 messages expected)
        log.info("Receiving initial data from server...")
        initial_data_list = []
        try:
            for i in range(6):
                data = comm.recv()
                log.debug(f"Received initial data part {i+1}/6: type {type(data)}")
                initial_data_list.append(data)

        except (ConnectionError, TimeoutError, ValueError) as e:
            log.critical(f"Failed to receive initial data: {e}. Exiting.")
            exit(0)  # Cannot proceed without initial data
        except Exception as e:
            log.critical(f"Unexpected error receiving initial data: {e}. Exiting.")
            exit(0)

        # 1. Weights
        weights = initial_data_list[0].copy()
        grad: np.ndarray = np.zeros_like(
            weights, dtype=np.float32
        )  # Ensure grad is float32
        log.info(f"Received initial weights, shape: {weights.shape}")

        # 2. Tokens
        tokens = initial_data_list[1]
        # 3. Token info (lengths)
        token_info = initial_data_list[2]  # Should be list of lengths

        # Reconstruct batchTokens from tokens and info
        batchTokens = []
        if isinstance(token_info, list) and token_info:
            start_idx = 0
            for length in token_info:
                batchTokens.append(tokens[start_idx : start_idx + length])
                start_idx += length
            totalNumTokens = sum(token_info)
            maxNumTokens = float(max(token_info)) if token_info else 0.0
            log.info(
                f"Received initial tokens: {len(batchTokens)} batches, {totalNumTokens} total tokens."
            )
        else:
            totalNumTokens = 0
            maxNumTokens = 0.0
            log.warning("Received empty or invalid initial token info.")

        # 4. Optimizer state
        optimizer_values = initial_data_list[3]
        log.info(f"Received optimizer state, shape: {optimizer_values.shape}")

        # 5. Config
        config = initial_data_list[4]  # Should be a dict
        if not isinstance(config, dict):
            log.critical("Received invalid config data. Exiting.")
            exit(0)
        log.info(f"Received config: {config}")

        # 6. Random seed
        seed = initial_data_list[5]  # Should be an int/uint
        log.info(f"Received initial seed: {seed}")

        # Initialize optimizer
        log.info("Initializing optimizer...")
        optimizer = AdamOptimizer(
            weights.shape[0],
            config["learningRate"],
            config["beta1"],
            config["beta2"],
        )
        # Ensure optimizer values have correct shape before assignment
        if optimizer_values.shape[0] == 2 * weights.shape[0]:
            optimizer.m = optimizer_values[: weights.shape[0]].astype(np.float32)
            optimizer.v = optimizer_values[weights.shape[0] :].astype(np.float32)
            optimizer.t = config["stepNum"]
            # Precompute powers (important!)
            optimizer.beta1Power = config["beta1"] ** (optimizer.t + 1)
            optimizer.beta2Power = config["beta2"] ** (optimizer.t + 1)
            log.info(f"Optimizer initialized at step {optimizer.t}")
        else:
            log.critical(
                f"Optimizer state shape mismatch. Expected {2*weights.shape[0]}, got {optimizer_values.shape[0]}. Exiting."
            )
            exit(0)

        log.info("Initializing model...")
        # Use modelType from config if available, otherwise default
        model_type = config.get("modelType", "chat")
        if model_type == "chat":
            model = ChatModel()
        else:
            log.warning(f"Unknown model type '{model_type}', using ChatModel.")
            model = ChatModel()  # Default or raise error

        firstStep = True

        while comm.is_connected():
            log.info(
                f"\n--- Waiting for Server Instructions (Step {optimizer.t + 1}) ---"
            )

            # Receive weight request or continuation signal
            log.info(f"Checking for weights request...")
            start_recv = perf_counter()
            try:
                request = comm.recv()  # Expect "need weights" or "dont need weights"
                elapsed_recv = perf_counter() - start_recv
                log.info(f"Received instruction '{request}' ({elapsed_recv:.4f}s)")

                if request == "need weights":
                    log.info("Server requested weights. Sending...")
                    # Prepare data: step num, m, v, weights
                    data_to_send = np.concatenate(
                        [
                            np.array(
                                [optimizer.t], dtype=np.float32
                            ),  # Send step number as float32
                            optimizer.m,
                            optimizer.v,
                            weights,
                        ]
                    ).astype(
                        np.float32
                    )  # Ensure final type
                    comm.send(data_to_send)
                    log.info(f"Sent weights to server.")

                    # Server will always send "dont need weights" after handling the request
                    # (or disconnect if something went wrong server-side)
                    follow_up = comm.recv()
                    log.info(f"Received follow-up '{follow_up}'")
                    if follow_up != "dont need weights":
                        log.warning(
                            f"Expected 'dont need weights' after sending, got '{follow_up}'"
                        )

                elif request == "dont need weights":
                    # Normal continuation
                    pass
                else:
                    log.warning(f"Received unexpected instruction: {request}")
                    # Continue assuming it's a normal step, but log warning

            except (ConnectionError, TimeoutError, ValueError) as e:
                log.error(f"Communication error receiving instruction: {e}")
                break  # Exit loop if communication fails
            except Exception as e:
                log.error(f"Unexpected error receiving instruction: {e}")
                break

            # Receieve new tokens/indicator
            log.info("Receiving new tokens (or indicator)...")
            try:
                # Expect 2 messages: token bytes and token info list
                tokens_data = comm.recv()
                token_info_new = comm.recv()

                # Update batchTokens only if we received actual new tokens
                if isinstance(token_info_new, list) and token_info_new:
                    log.info(f"Received {len(token_info_new)} new token batches.")
                    batchTokens = []
                    tokens_np = np.frombuffer(
                        tokens_data, dtype=np.uint16
                    )  # Assuming uint16
                    start_idx = 0
                    for length in token_info_new:
                        batchTokens.append(tokens_np[start_idx : start_idx + length])
                        start_idx += length
                    totalNumTokens = sum(token_info_new)
                    maxNumTokens = float(max(token_info_new)) if token_info_new else 0.0
                    log.info(
                        f"Updated tokens: {len(batchTokens)} batches, {totalNumTokens} total tokens."
                    )
                elif (
                    isinstance(tokens_data, np.ndarray)
                    and tokens_data.size == 0
                    and token_info_new == []
                ):
                    log.info("No new tokens received, using previous batch.")
                    # Keep using existing batchTokens
                else:
                    log.warning(
                        f"Received unexpected token data format. Types: {type(tokens_data)}, {type(token_info_new)}. Keeping old tokens."
                    )

            except (ConnectionError, TimeoutError, ValueError) as e:
                log.error(f"Communication error receiving tokens: {e}")
                break  # Exit loop
            except Exception as e:
                log.error(f"Unexpected error receiving tokens: {e}")
                break

            # Check if we have any tokens to process
            if not batchTokens:
                log.warning("No tokens available to process for this step. Waiting.")
                # Need to decide how to handle this - maybe send empty rewards?
                # For now, let's send a dummy reward array [seed, nan]
                try:
                    dummy_rewards = np.array([seed, np.nan], dtype=np.float32)
                    comm.send(dummy_rewards)
                    log.info("Sent dummy rewards as no tokens were available.")
                    # Then try to receive the normalized rewards/info to stay in sync
                    norm_rewards_bytes = comm.recv()  # bytes
                    response_data = comm.recv()  # dict
                    seed = response_data["seed"]  # Update seed for next step
                    stepStart = perf_counter()  # Reset step timer
                    firstStep = False  # Mark first step as done
                    continue  # Skip weight update as no work was done
                except (ConnectionError, TimeoutError, ValueError) as e:
                    log.error(f"Communication error during dummy reward cycle: {e}")
                    break

            # Run trials
            trialStart = perf_counter()
            time_limit = config["timePerStep"]
            log.info(
                f"Running trials for ~{time_limit}s on {totalNumTokens} tokens. Avg. batch size: {totalNumTokens / maxNumTokens if maxNumTokens > 0 else 0:.2f}"
            )
            rewards = [seed]  # Start rewards list with the seed used for this step
            numTrials = 0
            np.random.seed(seed)  # Seed RNG for this step's perturbations

            # Main trial loop: check time before starting next trial
            while True:
                # Estimate time for the *next* trial
                time_taken_so_far = perf_counter() - trialStart
                estimated_time_next_trial = (
                    timePerTrial if numTrials > 0 else 0.5
                )  # Use estimate based on previous iterations or a default guess if this is the first iter
                estimated_end_time = (
                    perf_counter() - stepStart + estimated_time_next_trial
                )

                # Break if adding another trial would exceed the time limit
                # Add a small buffer (e.g., 0.1s) to be safe
                if not firstStep and estimated_end_time > (time_limit - 0.1):
                    log.debug(
                        f"Stopping trials: Est. end time {estimated_end_time:.2f}s >= limit {time_limit - 0.1:.2f}s"
                    )
                    break

                # Run one trial
                perturbation = (
                    np.random.randn(weights.shape[0]).astype(np.float32)
                    * config["sigma"]
                )
                loss = model.getLossBatched(
                    weights + perturbation,
                    batchTokens,
                    config["hiddenSize"],
                    config["vocabSize"],
                    config["nLayers"],
                )
                rewards.append(loss)
                numTrials += 1

                # If it was the first step, only run one trial
                if firstStep:
                    log.info("First step: running only one trial.")
                    break

                # Update average time per trial *after* the trial completes
                timePerTrial = (perf_counter() - trialStart) / numTrials

                # Safety break if loop runs unexpectedly long (e.g., time estimate is wrong)
                if (perf_counter() - stepStart) > (time_limit):
                    log.warning("Stopping trials: Exceeded time limit.")
                    break

            trialEnd = perf_counter()
            elapsed = trialEnd - trialStart
            # Update timePerTrial based on actual time taken
            timePerTrial = (
                elapsed / numTrials if numTrials > 0 else timePerTrial
            )  # Keep old estimate if no trials ran

            log.info(
                f"Completed {numTrials} trials in {elapsed:.4f}s. Avg time/trial: {timePerTrial:.4f}s"
            )
            if elapsed > 0:
                tokens_per_sec = (numTrials * totalNumTokens) / elapsed
                log.info(f"Throughput: {tokens_per_sec:,.2f} tok/sec")

            # Send rewards back to server
            try:
                rewards_np = np.array(rewards).astype(np.float32)
                log.info(f"Sending {len(rewards_np)-1} rewards to server.")
                comm.send(rewards_np)
            except (ConnectionError, TimeoutError) as e:
                log.error(f"Communication error sending rewards: {e}")
                break  # Exit loop

            # Receive normalized rewards and next step info
            log.info("Waiting for normalized rewards and next seed...")
            try:
                # Receive rewards (as bytes)
                normalizedRewardsBytes = comm.recv()
                # Receive response dict (reward_info, seed)
                response_data = comm.recv()  # dict

                # Decode normalized rewards from bytes
                normalizedRewards = np.frombuffer(
                    normalizedRewardsBytes, dtype=np.float32
                )
                reward_info = response_data["reward_info"]  # List of (nTrials, seed)
                next_seed = response_data["seed"]  # Seed for the *next* step

                log.info(f"Received {len(normalizedRewards)} normalized rewards.")
                log.info(
                    f"Received reward info for {len(reward_info)} clients/batches."
                )
                log.info(f"Next seed: {next_seed}")

            except (ConnectionError, TimeoutError, ValueError) as e:
                log.error(f"Communication error receiving normalized rewards: {e}")
                break  # Exit loop
            except KeyError as e:
                log.error(
                    f"Received invalid response data (missing key {e}): {response_data}"
                )
                break
            except Exception as e:
                log.error(f"Unexpected error receiving normalized rewards: {e}")
                break

            # --- Weight Update ---
            update_start = perf_counter()
            grad.fill(0)  # Reset gradient

            total_rewards_processed = 0
            total_rewards_expected_from_info = sum(item[0] for item in reward_info)

            if len(normalizedRewards) != total_rewards_expected_from_info:
                log.warning(
                    f"Mismatch between normalized rewards received ({len(normalizedRewards)}) and expected from info ({total_rewards_expected_from_info}). Update might be incorrect."
                )
                # Decide how to handle: skip update? use min length? For now, proceed cautiously.

            processed_reward_idx = 0
            for nTrials_from_info, trialSeed_from_info in reward_info:
                if processed_reward_idx >= len(normalizedRewards):
                    log.warning(
                        "Ran out of normalized rewards while processing reward_info. Stopping gradient accumulation early."
                    )
                    break

                np.random.seed(
                    trialSeed_from_info
                )  # Ensure RNG matches the step where rewards were generated
                log.debug(
                    f"Processing {nTrials_from_info} rewards using seed {trialSeed_from_info}"
                )

                for _ in range(nTrials_from_info):
                    if processed_reward_idx >= len(normalizedRewards):
                        # This check prevents index error if counts mismatch
                        log.warning(
                            "Reward count mismatch during inner loop. Stopping gradient accumulation."
                        )
                        break  # Break inner loop

                    # Generate the *same* perturbation used for this reward
                    perturbation = np.random.randn(weights.shape[0]).astype(
                        np.float32
                    )  # Match type
                    # Add weighted perturbation to gradient
                    grad += perturbation * normalizedRewards[processed_reward_idx]
                    processed_reward_idx += 1
                else:
                    continue  # Only executed if the inner loop completes without break
                break  # Executed if the inner loop breaks

            log.info(f"Accumulated gradient from {processed_reward_idx} rewards.")

            # Apply optimizer update
            if config["optimizer"] == "adam" and processed_reward_idx > 0:
                update_vector = optimizer.getGrad(
                    grad
                )  # Adam calculates the final update step
            elif config["optimizer"] == "sgd" and processed_reward_idx > 0:
                update_vector = grad * config["learningRate"]  # Simple SGD scaling
            else:
                update_vector = np.zeros_like(
                    grad
                )  # No update if no rewards or unknown optimizer
                if processed_reward_idx > 0:
                    log.warning(
                        f"Unknown optimizer '{config['optimizer']}'. No weight update performed."
                    )

            weights -= update_vector  # Apply update
            gradNorm = np.sqrt(
                np.sum(update_vector**2)
            )  # Calculate norm of the actual update applied

            update_elapsed = perf_counter() - update_start
            log.info(
                f"Updated weights ({update_elapsed:.4f}s). Grad Norm (applied update norm): {gradNorm:.6f}"
            )

            # Prepare for next step
            seed = next_seed  # Use the seed received from the server for the next iteration
            stepStart = perf_counter()  # Reset timer for the next step
            firstStep = False  # No longer the first step

    # End of loop (connection lost or error)
    except Exception as e:
        # Catch any unexpected errors in the main loop
        log.critical(
            f"An unexpected error occurred in the client main loop: {e}", exc_info=True
        )  # Log traceback
    finally:
        log.info("Closing connection.")
        comm.close()
        log.info("Client finished.")
