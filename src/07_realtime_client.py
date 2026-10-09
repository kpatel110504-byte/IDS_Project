"""07_realtime_client.py
Very simple client to talk to the IDS server.
"""
import os
import json
import socket
import numpy as np

MODELS_DIR = os.path.join("results","models")
HOST = "127.0.0.1"
PORT = 5005

def load_example_features():
    X_test = np.load(os.path.join(MODELS_DIR,"X_test.npy"))
    return X_test[0].tolist()

def main():
    features = load_example_features()
    payload = {"features": features}
    msg = json.dumps(payload)
    print("Connecting to server...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("Sending features...")
        s.sendall(msg.encode("utf-8"))
        data = s.recv(65536)
    resp = data.decode("utf-8")
    print("Raw server response:", resp)
    response = json.loads(resp)
    print("\n=== Server Prediction ===")
    if "error" in response:
        print("Error:", response["error"])
    else:
        print("Prediction:", response["prediction"])
        print("Meaning   :", response["meaning"])
        print("Probability:", response["probability"])

if __name__ == "__main__":
    main()
