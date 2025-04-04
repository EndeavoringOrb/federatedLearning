import socket
import struct
import pickle
import numpy as np
from datetime import datetime
import sys
import logging
from functools import wraps
from typing import Optional, Any, Tuple, Union


# Configure logging
class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%H:%M:%S.%f")[:-3]


class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        sys.stdout.flush()


# Configure logging with a custom formatter
handler = FlushHandler()
formatter = MicrosecondFormatter("%(asctime)s|%(levelname)s|%(message)s")
handler.setFormatter(formatter)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)
#log.setLevel(logging.DEBUG)  # Uncomment for verbose logging

# Constants
HEADER_FORMAT = "II"  # Dtype Length, Data Length
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
BUFFER_SIZE = 4096  # Receive buffer size


def socket_operation(func):
    """Decorator for socket operations with error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except socket.timeout:
            log.error(f"Socket timeout during {func.__name__}")
            raise TimeoutError("Socket timed out")
        except OSError as e:
            log.error(f"Socket error during {func.__name__}: {e}")
            raise ConnectionError(f"Socket error: {e}")
        except Exception as e:
            log.error(f"Unexpected error during {func.__name__}: {e}")
            raise

    return wrapper


@socket_operation
def recvall(sock: socket.socket, size: int) -> bytes:
    """Helper function to receive exactly 'size' bytes from a socket."""
    received_chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, BUFFER_SIZE))
        if not chunk:
            log.error(f"Socket connection broken while receiving data.")
            raise ConnectionError("Socket connection broken")
        received_chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(received_chunks)


class CommunicationTarget:
    """Simple class to hold IP and Port."""

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def __repr__(self) -> str:
        return f"{self.ip}:{self.port}"

    def get(self) -> Tuple[str, int]:
        return (self.ip, self.port)


class CommunicationHandler:
    """Handles communication, usable as an instance (client) or via static methods (server)."""

    def __init__(self, ip: Optional[str] = None, port: Optional[int] = None):
        self.target_socket = None
        self.target_info = None
        self.connected = False
        if ip and port:
            self.target_info = CommunicationTarget(ip, port)

    @socket_operation
    def connect(self) -> bool:
        """Establishes connection for the client instance."""
        if not self.target_info:
            raise ValueError("Cannot connect without target IP and port.")
        if self.connected:
            log.warning("Already connected.")
            return True

        self.target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set TCP keepalive options
        self.target_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Platform-specific keepalive options could be added here
        # self.target_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)  # Linux
        # self.target_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 3)  # Linux
        # self.target_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)  # Linux

        self.target_socket.connect(self.target_info.get())
        self.connected = True
        log.info(f"Connected to {self.target_info}")
        return True

    def close(self) -> None:
        """Closes the connection for the client instance."""
        if self.connected and self.target_socket:
            try:
                self.target_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # Ignore if already closed
            self.target_socket.close()
            self.connected = False
            log.info(f"Connection to {self.target_info} closed.")
        self.target_socket = None

    def is_connected(self) -> bool:
        """Checks if the client instance is connected."""
        return self.connected and self.target_socket is not None

    @staticmethod
    @socket_operation
    def _static_send(sock: socket.socket, data_to_send: Any) -> None:
        """
        Sends data over the given socket, determining type automatically.
        Prepends header with dtype length and data length.
        """
        # Determine data type and convert to bytes
        if isinstance(data_to_send, np.ndarray):
            data_bytes = data_to_send.tobytes()
            dtype = f"np.{data_to_send.dtype}"
        elif isinstance(data_to_send, str):
            data_bytes = data_to_send.encode("utf-8")
            dtype = "str"
        elif isinstance(data_to_send, bytes):
            data_bytes = data_to_send
            dtype = "bytes"
        else:
            # Default to pickle for other types
            try:
                data_bytes = pickle.dumps(data_to_send)
                dtype = "pickle"
            except (pickle.PicklingError, TypeError) as e:
                log.error(f"Could not serialize data: {e}")
                raise TypeError(f"Data serialization failed: {e}")

        # Encode type information
        dtype_encoded = dtype.encode("utf-8")
        dtype_len = len(dtype_encoded)
        data_len = len(data_bytes)

        # Pack header and send
        header = struct.pack(HEADER_FORMAT, dtype_len, data_len)
        log.debug(
            f"SENDING data: dtype='{dtype}', dtype_len={dtype_len}, data_len={data_len}"
        )

        # Send data in parts: header, dtype, data
        sock.sendall(header)
        sock.sendall(dtype_encoded)
        sock.sendall(data_bytes)
        log.debug(f"Sent {data_len} bytes successfully.")

    @staticmethod
    @socket_operation
    def _static_recv(sock: socket.socket) -> Any:
        """
        Receives data from the given socket.
        Reads header, determines type, and decodes the data.
        """
        # Receive header
        header_bytes = recvall(sock, HEADER_SIZE)
        dtype_len, data_len = struct.unpack(HEADER_FORMAT, header_bytes)
        log.debug(f"RECEIVING data: dtype_len={dtype_len}, data_len={data_len}")

        # Receive dtype string
        dtype_encoded = recvall(sock, dtype_len)
        dtype = dtype_encoded.decode("utf-8")
        log.debug(f"Received dtype: '{dtype}'")

        # Receive actual data
        data_bytes = recvall(sock, data_len)
        log.debug(f"Received {len(data_bytes)} bytes successfully.")

        # Decode data based on dtype
        try:
            if dtype.startswith("np."):
                np_dtype = getattr(np, dtype[3:])  # Get numpy dtype (e.g., np.float32)
                return np.frombuffer(data_bytes, dtype=np_dtype)
            elif dtype == "str":
                return data_bytes.decode("utf-8")
            elif dtype == "bytes":
                return data_bytes  # Return as raw bytes
            elif dtype == "pickle":
                return pickle.loads(data_bytes)
            else:
                log.warning(f"Unknown dtype '{dtype}' received. Returning raw bytes.")
                return data_bytes  # Fallback to raw bytes for unknown types
        except (AttributeError, TypeError) as e:
            log.error(f"Error decoding data with dtype '{dtype}': {e}")
            raise ValueError(f"Invalid data type '{dtype}' received")
        except pickle.UnpicklingError as e:
            log.error(f"Error unpickling data: {e}")
            raise ValueError("Failed to unpickle received data")

    def send(self, data_to_send: Any) -> None:
        """Instance method to send data over the managed socket."""
        if not self.is_connected():
            log.error("Cannot send: Not connected.")
            raise ConnectionError("Not connected")
        try:
            CommunicationHandler._static_send(self.target_socket, data_to_send)
        except (TimeoutError, ConnectionError) as e:
            log.error(f"Send failed for {self.target_info}: {e}. Closing connection.")
            self.close()  # Close connection on send failure
            raise e

    def recv(self) -> Any:
        """Instance method to receive data from the managed socket."""
        if not self.is_connected():
            log.error("Cannot receive: Not connected.")
            raise ConnectionError("Not connected")
        try:
            return CommunicationHandler._static_recv(self.target_socket)
        except (TimeoutError, ConnectionError, ValueError) as e:
            log.error(
                f"Receive failed for {self.target_info}: {e}. Closing connection."
            )
            self.close()  # Close connection on receive failure
            raise e

    @staticmethod
    def send_message(sock: socket.socket, data_to_send: Any) -> None:
        """Convenience static method for sending."""
        CommunicationHandler._static_send(sock, data_to_send)

    @staticmethod
    def receive_message(sock: socket.socket) -> Any:
        """Convenience static method for receiving."""
        return CommunicationHandler._static_recv(sock)
