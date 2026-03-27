"""Startup script: wait for services, pull model, launch Streamlit."""

import os
import sys
import time
import socket
import subprocess
import requests


def wait_for_mariadb(host, port, timeout=60):
    """Wait for MariaDB to accept TCP connections."""
    print(f"Väntar på MariaDB ({host}:{port})...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.create_connection((host, port), timeout=2)
            sock.close()
            # Also try a real connection
            import mariadb
            conn = mariadb.connect(
                host=host, port=port,
                user=os.getenv("DB_USER", "bookrag"),
                password=os.getenv("DB_PASSWORD", "bookrag_pass"),
                database=os.getenv("DB_NAME", "bookrag"),
            )
            conn.cursor().execute("SELECT 1")
            conn.close()
            print("MariaDB redo.")
            return True
        except Exception:
            time.sleep(2)
    print("TIMEOUT: MariaDB svarade inte inom 60s.")
    return False


def wait_for_ollama(base_url, timeout=60):
    """Wait for Ollama API to respond."""
    print(f"Väntar på Ollama ({base_url})...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                print("Ollama redo.")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("TIMEOUT: Ollama svarade inte inom 60s.")
    return False


def ensure_model(base_url, model_name):
    """Check if model exists, pull if not."""
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    models = [m["name"] for m in resp.json().get("models", [])]
    if model_name in models:
        print(f"Modell {model_name} finns redan.")
        return
    # Also check without tag
    base_name = model_name.split(":")[0]
    if any(m.startswith(base_name) for m in models):
        print(f"Modell {model_name} finns redan.")
        return
    print(f"Laddar ner modell {model_name}... (detta kan ta några minuter)")
    resp = requests.post(
        f"{base_url}/api/pull",
        json={"name": model_name, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    print(f"Modell {model_name} nedladdad.")


def main():
    db_host = os.getenv("DB_HOST", "db")
    db_port = int(os.getenv("DB_PORT", "3306"))
    ollama_host = os.getenv("OLLAMA_HOST", "ollama")
    ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
    ollama_url = f"http://{ollama_host}:{ollama_port}"

    if not wait_for_mariadb(db_host, db_port):
        sys.exit(1)

    if not wait_for_ollama(ollama_url):
        sys.exit(1)

    ensure_model(ollama_url, "nomic-embed-text:latest")

    print("Startar Streamlit...")
    os.execvp("streamlit", [
        "streamlit", "run", "/app/src/app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
    ])


if __name__ == "__main__":
    main()
