import socket
from basicCommunicationUtils import *

def start_client():
    # Server settings
    server_ip = "130.215.211.30"
    server_port = 55551
    
    # Create a socket object
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    
    connected = True
    while connected:
        # Send a message to the server
        message = input("Enter message to send: ").encode("utf-8")
        message = np.random.randn(10000).astype(np.float32).tobytes()
        sendBytes(client, message, "SERVER")

        # Receive response from the server
        response, valid = receiveData(client, "text", "SERVER")
        if not valid:
            connected = False
        else:
            print(f"[SERVER] {response}")
    
    client.close()

if __name__ == "__main__":
    start_client()
