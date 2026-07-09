"""
=============================================================
FLASK WEB APP — CHATBOT AKADEMIK ITI
Jalankan: python app.py
Buka browser: http://localhost:5000
=============================================================
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import json
import re
import random
import os

app = Flask(__name__)

# ── Preprocessing (sama dengan train.py) ──────────────────────
PREFIXES  = ['me', 'di', 'ke', 'se', 'ter', 'ber', 'pe', 'per', 'meng',
             'men', 'mem', 'peng', 'pen', 'pem', 'meny', 'peny']
SUFFIXES  = ['kan', 'an', 'i', 'nya', 'lah', 'kah', 'pun']
STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'dalam', 'tidak', 'juga', 'sudah', 'ada', 'saya',
    'kamu', 'anda', 'atau', 'akan', 'bisa', 'apa', 'bagaimana', 'cara',
    'kalau', 'jika', 'mau', 'maka', 'tapi', 'tetapi', 'sudah', 'belum'
}

def stem(word):
    for pref in PREFIXES:
        if word.startswith(pref) and len(word) > len(pref) + 2:
            word = word[len(pref):]
            break
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            word = word[:-len(suf)]
            break
    return word

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]', '', sentence)
    return sentence.split()

def preprocess(sentence):
    tokens = tokenize(sentence)
    tokens = [stem(w) for w in tokens if w not in STOPWORDS and len(w) > 1]
    return tokens

def bag_of_words(tokens, vocab):
    bag = np.zeros(len(vocab), dtype=np.float32)
    for i, word in enumerate(vocab):
        if word in tokens:
            bag[i] = 1.0
    return bag

# ── Model ANN ─────────────────────────────────────────────────
class ChatbotANN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ── Load model & dataset ───────────────────────────────────────
DEVICE = torch.device('cpu')

def load_model():
    data      = torch.load('chatbot_model.pth', map_location=DEVICE)
    model     = ChatbotANN(
        data['input_size'], data['hidden1'],
        data['hidden2'],    data['output_size'],
        data['dropout']
    ).to(DEVICE)
    model.load_state_dict(data['model_state'])
    model.eval()
    return model, data['vocab'], data['tags']

with open('dataset.json', 'r', encoding='utf-8') as f:
    DATASET = json.load(f)

MODEL, VOCAB, TAGS = load_model()
print("✅ Model chatbot berhasil dimuat!")

# ── Fungsi prediksi ────────────────────────────────────────────
def predict_intent(text, threshold=0.5):
    """
    Memprediksi intent dari input teks pengguna.
    Return: (intent_tag, confidence, response)
    """
    tokens = preprocess(text)
    bow    = bag_of_words(tokens, VOCAB)
    x      = torch.tensor([bow], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)[0]
        probs  = torch.softmax(logits, dim=0).numpy()

    max_prob = float(np.max(probs))
    max_idx  = int(np.argmax(probs))
    tag      = TAGS[max_idx]

    if max_prob < threshold:
        return "unknown", max_prob, None

    # Ambil respons acak dari intent yang diprediksi
    for intent in DATASET['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return tag, max_prob, response

    return "unknown", max_prob, None

# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': 'Mohon masukkan pertanyaan Anda.', 'intent': '', 'confidence': 0})

    tag, confidence, response = predict_intent(user_message)

    if tag == "unknown" or response is None:
        response = (
            "Maaf, saya belum memahami pertanyaan tersebut. "
            "Coba tanyakan seputar: KRS, jadwal kuliah, nilai, tugas akhir, "
            "lokasi kampus, pembayaran, dosen, cuti, atau wisuda."
        )

    return jsonify({
        'response':   response,
        'intent':     tag,
        'confidence': round(confidence * 100, 2)
    })

@app.route('/intents')
def get_intents():
    """Endpoint untuk melihat daftar intent yang tersedia."""
    intent_list = [
        {'tag': i['tag'], 'patterns_count': len(i['patterns'])}
        for i in DATASET['intents']
    ]
    return jsonify({'intents': intent_list, 'total': len(intent_list)})

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("   CHATBOT AKADEMIK ITI — TEKNIK INFORMATIKA")
    print("=" * 50)
    print("   Buka browser: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
