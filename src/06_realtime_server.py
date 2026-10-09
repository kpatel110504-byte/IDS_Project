"""06_realtime_server.py
Very simple socket server for real-time IDS prediction.
"""
import os
import json
import socket
import numpy as np
import joblib

MODELS_DIR = os.path.join("results","models")
HOST = "127.0.0.1"
PORT = 5005

def load_model():
    model = joblib.load(os.path.join(MODELS_DIR,"random_forest_model.joblib"))
    n_features = model.n_features_in_
    return model, n_features

def handle_client(conn, addr, model, n_features):
    print("Connected by", addr)
    data = conn.recv(65536)
    if not data:
        print("No data from client.")
        return
    msg = data.decode("utf-8")
    print("Message from client:", msg)
    payload = json.loads(msg)
    features = payload.get("features", [])
    x = np.array(features, dtype="float32").reshape(1, -1)
    if x.shape[1] != n_features:
        response = {"error": f"Expected {n_features} features, got {x.shape[1]}"}
    else:
        pred = model.predict(x)[0]
        if hasattr(model,"predict_proba"):
            proba = float(model.predict_proba(x)[0, int(pred)])
        else:
            proba = None
        response = {
            "prediction": int(pred),
            "meaning": "0 = normal, 1 = attack",
            "probability": proba,
        }
    conn.sendall(json.dumps(response).encode("utf-8"))
    print("Response sent:", response)

def run_server():
    model, n_features = load_model()
    print("Server started on", f"{HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        while True:
            print("Waiting for client...")
            conn, addr = s.accept()
            with conn:
                handle_client(conn, addr, model, n_features)

if __name__ == "__main__":
    run_server()
