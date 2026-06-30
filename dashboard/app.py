from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)
SERVER_URL = "http://127.0.0.1:3000"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    try:
        resp = requests.get(f"{SERVER_URL}/status", timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.RequestException as exc:
        return jsonify({
            "error": "failed to fetch server status",
            "detail": str(exc)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
