import socket
import struct
import pickle
import numpy as np

headerFormat = "I"
headerSize = 4
DEBUG = True
BUFFER_SIZE = 1024


def sendBytes(connection: socket.socket, data: bytes, addr):
    if DEBUG:
        print(f"SENDING DATA")
    dataLength = len(data)

    # send header
    connection.send(struct.pack(headerFormat, dataLength))
    if DEBUG:
        print(f"  Sent header specifying length of {dataLength} bytes")

    # chunk data
    if DEBUG:
        print(f"  Chunking bytes")
    dataChunks = []
    while data:
        dataChunks.append(data[:BUFFER_SIZE])
        data = data[BUFFER_SIZE:]
    numChunks = len(dataChunks)

    # send chunks
    if DEBUG:
        print(f"  Sending chunks")
    for i, chunk in enumerate(dataChunks):
        connection.send(chunk)
        if DEBUG:
            print(f"  Sent chunk {i+1}/{numChunks} with length {len(chunk)}")

    # see if chunks were properly received
    msg = connection.recv(headerSize)
    msgLength = struct.unpack(headerFormat, msg)[0]
    msg = connection.recv(msgLength)
    properlyReceived = pickle.loads(msg) == "chunksGood"
    if DEBUG:
        if properlyReceived:
            print(f"  Chunks were properly received")
        else:
            print(f"  Chunks were NOT properly received")

    # re-send stuff if not properly received
    while not properlyReceived:
        # receive list of failed chunks
        msg = connection.recv(headerSize)
        msgLength = struct.unpack(headerFormat, msg)[0]
        msg = connection.recv(msgLength)
        failedChunks = pickle.loads(msg)
        if DEBUG:
            print(f"  Got list of failed chunks")

        # re-send failed chunks
        if DEBUG:
            print(f"  Re-sending failed chunks")
        for chunkNum in failedChunks:
            connection.send(dataChunks[chunkNum])

        # see if chunks were properly received
        msg = connection.recv(headerSize)
        msgLength = struct.unpack(headerFormat, msg)[0]
        msg = connection.recv(msgLength)
        properlyReceived = pickle.loads(msg) == "chunksGood"

        if DEBUG:
            if properlyReceived:
                print(f"  Chunks were properly received")
            else:
                print(f"  Chunks were NOT properly received")


def receiveData(connection: socket.socket, dataType: str, addr):
    if DEBUG:
        print("RECEIVING DATA")
    msg = connection.recv(headerSize)
    if not msg:
        if DEBUG:
            print(f"  Received empty message from {addr}")
        return msg, False

    msgLength = struct.unpack(headerFormat, msg)[0]
    if DEBUG:
        print(f"  Receiving message with length {msgLength}")

    # Receive message bytes in chunks
    messages = []
    numChunks = (msgLength + BUFFER_SIZE - 1) // BUFFER_SIZE
    remainderBytes = msgLength - (msgLength // BUFFER_SIZE) * BUFFER_SIZE
    if DEBUG:
        print(f"Remainder bytes: {remainderBytes}")
    for i in range(numChunks):
        if i == numChunks - 1 and remainderBytes > 0:
            chunkLength = remainderBytes
        else:
            chunkLength = BUFFER_SIZE
        msg = connection.recv(chunkLength)
        messages.append(msg)
        if DEBUG:
            print(f"  Received chunk [{i+1}/{numChunks}] with length {len(msg)}/{chunkLength}")

    # Check for any chunks that were not received fully
    if DEBUG:
        print(f"  Validating chunks were fully received")
    failedChunks = []
    for i in range(numChunks - 1):
        if len(messages[i]) != BUFFER_SIZE:
            failedChunks.append((i, BUFFER_SIZE))
    if remainderBytes > 0 and len(messages[-1]) != remainderBytes:
        failedChunks.append((numChunks - 1, remainderBytes))

    # Send a message to say if all the data was properly received
    data = pickle.dumps("chunksBad" if len(failedChunks) > 0 else "chunksGood")
    dataLength = len(data)
    connection.send(struct.pack(headerFormat, dataLength))
    connection.send(data)

    while len(failedChunks) > 0:
        if DEBUG:
            print(
                f"  {len(failedChunks)} chunks not fully received, re-requesting chunks {failedChunks}"
            )
        # Re-request any chunks that failed
        data = pickle.dumps(
            [item[0] for item in failedChunks]
        )  # send the indices of the chunks that failed
        dataLength = len(data)
        connection.send(struct.pack(headerFormat, dataLength))
        connection.send(data)

        # Receive those chunks
        for i, (chunkNum, chunkLength) in enumerate(failedChunks):
            msg = connection.recv(chunkLength)
            messages[chunkNum] = msg
            if DEBUG:
                print(f"  Received re-requested chunk [{i+1}/{len(failedChunks)}]")

        # Check for any chunks that were not received fully
        if DEBUG:
            print(f"  Validating chunks were fully received")
        failedChunks = []
        for i in range(numChunks - 1):
            if len(messages[i]) != BUFFER_SIZE:
                failedChunks.append((i, BUFFER_SIZE))
        if remainderBytes > 0 and len(messages[-1]) != remainderBytes:
            failedChunks.append((numChunks - 1, remainderBytes))

        # Send a message to say if all the data was properly received
        data = pickle.dumps("chunksBad" if len(failedChunks) > 0 else "chunksGood")
        dataLength = len(data)
        connection.send(struct.pack(headerFormat, dataLength))
        connection.send(data)

    # Concatenate all messages
    msg = b""
    for message in messages:
        msg += message

    if DEBUG:
        print(f"  Received {len(msg)}/{msgLength} bytes in {numChunks} chunks")

    # decode based on datatype
    if dataType == "text":
        data = msg.decode("utf-8")
    elif dataType == "np.uint8":
        data = np.frombuffer(msg, np.uint8)
    elif dataType == "np.uint16":
        data = np.frombuffer(msg, np.uint16)
    elif dataType == "np.float32":
        data = np.frombuffer(msg, np.float32)
    elif dataType == "pickle":
        data = pickle.loads(msg)
    else:
        data = msg.decode("utf-8")  # just try normal decode

    # return decoded data
    return data, True
