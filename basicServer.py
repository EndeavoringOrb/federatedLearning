import socket
import threading
from threading import Lock
from basicCommunicationUtils import *
from utilitiesModel import *
from utilitiesMisc import *
from time import sleep, perf_counter
from shakespeareData import tokenLoader
import json
import os

config = {
    "timePerStep": 45,
    "learningRate": 1e-3,
    "sigma": 1e-3,
    "hiddenSize": 64,
    "vocabSize": 76,
    "nLayers": 3,
    "optimizer": "adam",  # "sgd" or "adam"
    "beta1": 0.9,
    "beta2": 0.999,
    "stepNum": 0,
    "modelType": "chat",
    "checkPointTime": 60,
    "newTokensInterval": 5,  # send new tokens every N steps
    "batchSize": 2,
}
seedHigh = 4_000_000


tokens = tokenLoader(config["vocabSize"], config["batchSize"])

if config["modelType"] == "critic":
    model = ChatCritic()
elif config["modelType"] == "minGru":
    model = MinGruChat()
else:
    model = ChatModel()

weights = model.getWeights(config["hiddenSize"], config["vocabSize"], config["nLayers"])
nParams = weights.shape[0]
optimizerValues = np.zeros(nParams * 2).astype(np.float32)
newWeights = False
all_rewards = []
reward_info = []
stepNum = 0

lastCheckPointTime = perf_counter()

clients = []
newClients = []

threadLock = Lock()


def getWeights(currentTrainingRun):
    global clients
    global config
    global stepNum
    global optimizerValues
    global weights
    global lastCheckPointTime

    print(f"Getting weights from {clients[0][1]}")
    # request weights
    sendBytes(clients[0][0], "need weights".encode("utf-8"), clients[0][1])

    # get weights
    data, valid = receiveData(clients[0][0], "np.float32", clients[0][1])
    if valid:
        print(f"[{clients[0][1]}] Sent weights")
        config["stepNum"] = int(data[0])
        stepNum = config["stepNum"]
        optimizerValues = data[1 : 1 + 2 * nParams]
        weights = data[1 + 2 * nParams :]
        np.save(
            f"trainingRuns/{currentTrainingRun}/model.npy",
            np.concatenate(
                (
                    [
                        config["hiddenSize"],
                        config["vocabSize"],
                        config["nLayers"],
                    ],
                    weights,
                )
            ),
        )
        lastCheckPointTime = perf_counter()
        return True
    else:
        print(f"Failed to get weights")
        return False


def handleClients():
    global clients
    global newClients
    global weights
    global optimizerValues
    global config
    global stepNum

    if len(os.listdir("trainingRuns")) == 0:
        currentTrainingRun = 0
    else:
        currentTrainingRun = max([int(item) for item in os.listdir("trainingRuns")]) + 1
    os.makedirs(f"trainingRuns/{currentTrainingRun}")

    with open(
        f"trainingRuns/{currentTrainingRun}/config.json", "w", encoding="utf-8"
    ) as f:
        json.dump(config, f)

    while True:
        # wait for some clients
        while len(clients) + len(newClients) == 0:
            sleep(0.05)

        stepNum += 1

        # send info to new clients
        if len(newClients) > 0:
            gotWeights = False
            if len(clients) > 0:
                gotWeights = getWeights(currentTrainingRun)

            if len(clients) == 0 or gotWeights:
                print(f"Sending weights to {len(newClients)} new clients")
                clientRemoveList = []
                for client in newClients:
                    try:
                        # send weights
                        sendBytes(client[0], weights.tobytes(), client[1])
                        # send tokens
                        batchTokens, batchInfo = next(tokens)
                        sendBytes(client[0], batchTokens.tobytes(), client[1])
                        # send tokens info
                        sendBytes(client[0], pickle.dumps(batchInfo), client[1])
                        # send optimizer state
                        sendBytes(client[0], optimizerValues.tobytes(), client[1])
                        # send config
                        sendBytes(client[0], pickle.dumps(config), client[1])
                        # send random seed
                        sendBytes(
                            client[0],
                            pickle.dumps(np.random.randint(0, seedHigh)),
                            client[1],
                        )
                    except Exception as e:
                        print(f"[ERROR] (sending initial data) {e}")
                        clientRemoveList.append(client)
                for client in clientRemoveList:
                    client[0].close()
                    newClients.remove(client)
                clients.extend(newClients)
                newClients = []
        elif perf_counter() - lastCheckPointTime > config["checkPointTime"]:
            gotWeights = getWeights(currentTrainingRun)

        clientRemoveList = []
        for client in clients:
            try:
                sendBytes(client[0], "dont need weights".encode("utf-8"), client[1])
            except Exception as e:
                print(f"[ERROR] (sending dont need weights) {e}")
                clientRemoveList.append(client)
        for client in clientRemoveList:
            client[0].close()
            clients.remove(client)

        # send new tokens if it is time to
        clientRemoveList = []
        for client in clients:
            try:
                if stepNum % config["newTokensInterval"] == 0:
                    # send tokens
                    batchTokens, batchInfo = next(tokens)
                    sendBytes(client[0], batchTokens.tobytes(), client[1])
                    # send tokens info
                    sendBytes(client[0], pickle.dumps(batchInfo), client[1])
                else:
                    # send tokens
                    batchTokens, batchInfo = next(tokens)
                    sendBytes(
                        client[0], np.array([]).astype(np.uint16).tobytes(), client[1]
                    )
                    # send tokens info
                    sendBytes(client[0], pickle.dumps([]), client[1])
            except Exception as e:
                print(f"[ERROR] (sending new tokens) {e}")
                clientRemoveList.append(client)
        for client in clientRemoveList:
            client[0].close()
            clients.remove(client)

        # get rewards
        print(f"Waiting for rewards")
        all_rewards = []
        reward_info = []
        clientRemoveList = []
        for client in clients:
            try:
                rewards, valid = receiveData(client[0], "np.float32", client[1])
                if valid:
                    print(f"[{client[1]}] Sent {len(rewards) - 1:,} rewards")
                    seed = rewards[0].astype(np.uint32)
                    with threadLock:
                        all_rewards.extend(rewards[1:])
                        reward_info.append((len(rewards) - 1, seed))
                else:
                    print(f"[{client[1]}] Failed sending rewards")
            except Exception as e:
                print(f"[ERROR] (getting rewards) {e}")
                clientRemoveList.append(client)
        for client in clientRemoveList:
            client[0].close()
            clients.remove(client)

        # process rewards
        numRewards = len(all_rewards)
        print(f"Total # Rewards: {numRewards:,}")
        if numRewards == 1:
            normalizedRewards = np.zeros(1)
            mean = 0
            print(f"Mean Reward: {0}")
        else:
            normalizedRewards = np.array(all_rewards)
            mean = normalizedRewards.mean()
            print(f"Mean Reward: {mean}")
            if np.isnan(mean):
                normalizedRewards = np.zeros(1)
                reward_info = [(1, 0)]
                print(f"Setting mean to 0 because of nan value")
            else:
                normalizedRewards = (normalizedRewards - mean) / np.std(all_rewards)

        print()
        seeds = np.random.randint(0, seedHigh, len(clients))
        normalizedRewardsBytes = normalizedRewards.astype(np.float32).tobytes()
        clientRemoveList = []
        for i, client in enumerate(clients):
            response = {
                "reward_info": reward_info,
                "seed": seeds[i],
            }
            try:
                sendBytes(client[0], normalizedRewardsBytes, client[1])
                sendBytes(client[0], pickle.dumps(response), client[1])
            except Exception as e:
                print(f"[ERROR] (sending normalized rewards) {e}")
                clientRemoveList.append(client)
                continue
            print(f"Sent rewards and info [{i+1}/{len(clients)}] {currentTime()}")

        for client in clientRemoveList:
            client[0].close()
            clients.remove(client)

        with open(
            f"trainingRuns/{currentTrainingRun}/loss.txt", "a", encoding="utf-8"
        ) as f:
            f.write(f"{mean}\n")


def start_server():
    # Server settings
    server_ip = "0.0.0.0"
    server_port = 55551

    # Create a socket object
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((server_ip, server_port))

    # Listen for incoming connections
    server.listen(5)
    print(f"[LISTENING] Server is listening on {server_ip}:{server_port}")

    # Start handler thread
    thread = threading.Thread(target=handleClients, args=())
    thread.start()

    while True:
        # Accept a connection
        client_socket, addr = server.accept()
        client_socket.settimeout(config["timePerStep"] + 5)
        newClients.append([client_socket, addr])
        print(f"[ACTIVE CONNECTIONS] {len(clients) + len(newClients)}")


if __name__ == "__main__":
    start_server()
