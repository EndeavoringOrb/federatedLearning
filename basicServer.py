import socket
import threading
from basicCommunicationUtils import *


# Function to handle client connections
def handle_client(client_socket, addr):
    print(f"[NEW CONNECTION] {addr} connected.")

    connected = True
    while connected:
        try:
            data, valid = receiveData(client_socket, "np.float32", addr)
            if valid:
                print(f"[{addr}] {data}")
                sendBytes(client_socket, f"Received: {data}".encode("utf-8"), addr)
            else:
                connected = False
        except Exception as e:
            print(f"[ERROR] Connection with {addr} lost. {e}")
            connected = False

    client_socket.close()
    print(f"[DISCONNECTED] {addr} disconnected.")


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

    while True:
        # Accept a connection
        client_socket, addr = server.accept()
        # Create a new thread for each client
        thread = threading.Thread(target=handle_client, args=(client_socket, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


if __name__ == "__main__":
    start_server()
