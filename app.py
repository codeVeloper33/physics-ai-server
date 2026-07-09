from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL = "BabalolaEpo/physics-llama-3b"

@app.route("/")
def home():
    return "Physics AI Server is running!"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "")
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": f"### Instruction:\n{question}\n\n### Response:\n",
                "parameters": {
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
        )
        result = response.json()
        answer = result[0].get("generated_text", "Could not get answer.")
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
