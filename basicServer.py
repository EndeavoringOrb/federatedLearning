import socket
import threading
from threading import Lock
from basicCommunicationUtils import *
from utilitiesModel import *
from utilitiesMisc import *
from time import sleep, perf_counter
from shakespeareData import tokenLoader

config = {
    "timePerStep": 5,
    "learningRate": 0.01,
    "sigma": 0.1,
    "hiddenSize": 32,
    "vocabSize": 74,
    "beta1": 0.9,
    "beta2": 0.999,
    "stepNum": 0,
    "modelType": "chat",
    "checkPointTime": 60,
}
seedHigh = 4_000_000


tokens = tokenLoader(config["vocabSize"], True)
if config["modelType"] == "critic":
    model = ChatCritic()
else:
    model = ChatModel()
weights = model.getWeights(config["hiddenSize"], config["vocabSize"])
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


def getWeights():
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
            f"weights/model.npy",
            np.concatenate(
                (
                    [
                        config["hiddenSize"],
                        config["vocabSize"],
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
    global newClients
    global weights
    global optimizerValues
    global config
    global stepNum

    while True:
        # wait for some clients
        while len(clients) + len(newClients) == 0:
            sleep(0.1)

        # send info to new clients
        if len(newClients) > 0:
            gotWeights = False
            if len(clients) > 0:
                gotWeights = getWeights()

            if len(clients) == 0 or gotWeights:
                print(f"Sending weights to {len(newClients)} new clients")
                for client in newClients:
                    # send weights
                    sendBytes(client[0], weights.tobytes(), client[1])
                    # send tokens
                    sendBytes(client[0], next(tokens).tobytes(), client[1])
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
                clients.extend(newClients)
                newClients = []
        elif perf_counter() - lastCheckPointTime > config["checkPointTime"]:
            gotWeights = getWeights()

        clientRemoveList = []
        for client in clients:
            try:
                sendBytes(client[0], "dont need weights".encode("utf-8"), client[1])
            except Exception as e:
                print(f"[ERROR] (sending normalized rewards) {e}")
                clientRemoveList.append(client)
        for client in clientRemoveList:
            clients.remove(client)


        # get rewards
        print(f"Waiting for rewards")
        all_rewards = []
        reward_info = []
        for client in clients:
            rewards, valid = receiveData(client[0], "np.float32", client[1])
            if valid:
                print(f"[{client[1]}] Sent {len(rewards) - 1:,} rewards")
                seed = rewards[0].astype(np.uint32)
                with threadLock:
                    all_rewards.extend(rewards[1:])
                    reward_info.append((len(rewards), seed))
            else:
                print(f"[{client[1]}] Failed sending rewards")

        # process rewards
        numRewards = len(all_rewards)
        if numRewards == 1:
            normalizedRewards = np.zeros(1)
            print(f"Mean Reward: {0}")
        else:
            mean = np.mean(all_rewards)
            print(f"Mean Reward: {mean}")
            normalizedRewards = (np.array(all_rewards) - mean) / np.std(all_rewards)

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
            clients.remove(client)


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
        newClients.append([client_socket, addr])
        print(f"[ACTIVE CONNECTIONS] {len(clients) + len(newClients)}")


if __name__ == "__main__":
    start_server()
