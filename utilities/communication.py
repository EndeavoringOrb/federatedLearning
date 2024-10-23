import socket
import struct
import pickle
import numpy as np
from datetime import datetime

headerFormat = "I"
headerSize = 4
DEBUG = False
BUFFER_SIZE = 1024


def currentTime():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log(message):
    if DEBUG:
        print(f"{currentTime()}|{message}", flush=True)


def recvall(sock: socket.socket, size):
    received_chunks = []
    remaining = size
    while remaining > 0:
        received = sock.recv(min(remaining, BUFFER_SIZE))
        if not received:
            raise Exception("unexpected EOF")
        received_chunks.append(received)
        remaining -= len(received)
    return b"".join(received_chunks)


def sendBytes(connection: socket.socket, data: bytes, addr):
    log(f"SENDING DATA")
    dataLength = len(data)

    # send header
    connection.send(struct.pack(headerFormat, dataLength))
    log(f"  Sent header specifying length of {len(data)} bytes")

    # chunk data
    log(f"  Chunking bytes")
    dataChunks = []
    while data:
        dataChunks.append(data[:BUFFER_SIZE])
        data = data[BUFFER_SIZE:]
    numChunks = len(dataChunks)

    # send chunks
    log(f"  Sending chunks")
    for i, chunk in enumerate(dataChunks):
        connection.send(chunk)
        log(f"  Sent chunk {i+1}/{numChunks} with length {len(chunk)}")


def receiveData(connection: socket.socket, dataType: str, addr):
    log("RECEIVING DATA")
    msg = recvall(connection, headerSize)
    msgLength = struct.unpack(headerFormat, msg)[0]
    if not msg:
        log(f"  Received empty message from {addr}")
        return msg, False
    log(f"  Receiving message with length {msgLength}")

    # Receive message bytes in chunks
    msg = recvall(connection, msgLength)
    log(f"  Received {len(msg)}/{msgLength} bytes")

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
