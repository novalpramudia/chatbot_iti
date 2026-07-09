"""
=============================================================
TRAINING MODEL ANN UNTUK CHATBOT AKADEMIK ITI
Mata Kuliah: Pemrosesan Bahasa Natural / Tugas Akhir
Framework  : PyTorch
=============================================================

Pipeline:
  dataset.json → Preprocessing (tokenize, stem) →
  Bag of Words → ANN Training → Simpan model
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import re
import os

# ── Simple stemmer Bahasa Indonesia ───────────────────────────
PREFIXES  = ['me', 'di', 'ke', 'se', 'ter', 'ber', 'pe', 'per', 'meng',
             'men', 'mem', 'peng', 'pen', 'pem', 'meny', 'peny']
SUFFIXES  = ['kan', 'an', 'i', 'nya', 'lah', 'kah', 'pun']

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
    return [stem(w) for w in tokens]

# Stopwords Bahasa Indonesia
STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'dalam', 'tidak', 'juga', 'sudah', 'ada', 'saya',
    'kamu', 'anda', 'atau', 'akan', 'bisa', 'apa', 'bagaimana', 'cara',
    'kalau', 'jika', 'mau', 'maka', 'tapi', 'tetapi', 'sudah', 'belum'
}

def bag_of_words(tokens, vocab):
    """Mengubah list token menjadi vektor Bag of Words."""
    stemmed = [stem(w) for w in tokens]
    bag = np.zeros(len(vocab), dtype=np.float32)
    for i, word in enumerate(vocab):
        if word in stemmed:
            bag[i] = 1.0
    return bag

# ── Arsitektur Model ANN ───────────────────────────────────────
class ChatbotANN(nn.Module):
    """
    Arsitektur ANN untuk klasifikasi intent chatbot.
    
    Input  → Hidden1 (ReLU) → Dropout → Hidden2 (ReLU) → Dropout → Output
    """
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

# ── Dataset Class ──────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── MAIN TRAINING ──────────────────────────────────────────────
def train():
    print("=" * 60)
    print("   TRAINING CHATBOT ANN — Akademik ITI")
    print("=" * 60)

    # 1. Load dataset
    with open('dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    intents = data['intents']
    print(f"\n[1] Dataset dimuat: {len(intents)} intent")

    # 2. Preprocessing & bangun vocabulary
    all_words = []
    tags = []
    xy = []  # (tokens, tag)

    for intent in intents:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tokens = preprocess(pattern)
            tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
            all_words.extend(tokens)
            xy.append((tokens, tag))

    # Vocabulary unik, sorted
    vocab = sorted(set(all_words))
    tags  = sorted(set(tags))

    print(f"\n[2] Preprocessing selesai")
    print(f"    Vocabulary: {len(vocab)} kata")
    print(f"    Jumlah intent: {len(tags)}")
    print(f"    Total training samples: {len(xy)}")
    print(f"    Intent: {tags}")

    # 3. Buat data training (Bag of Words)
    X_train, y_train = [], []
    for (tokens, tag) in xy:
        bow = bag_of_words(tokens, vocab)
        X_train.append(bow)
        y_train.append(tags.index(tag))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"\n[3] Bag of Words shape: {X_train.shape}")

    # 4. Dataloader
    dataset    = IntentDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 5. Model
    INPUT_SIZE  = len(vocab)
    HIDDEN1     = 128
    HIDDEN2     = 64
    OUTPUT_SIZE = len(tags)
    DROPOUT     = 0.3
    EPOCHS      = 300
    LR          = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ChatbotANN(INPUT_SIZE, HIDDEN1, HIDDEN2, OUTPUT_SIZE, DROPOUT).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print(f"\n[4] Arsitektur Model ANN")
    print(f"    Input  : {INPUT_SIZE}")
    print(f"    Hidden1: {HIDDEN1} (ReLU + BatchNorm + Dropout)")
    print(f"    Hidden2: {HIDDEN2} (ReLU + BatchNorm + Dropout)")
    print(f"    Output : {OUTPUT_SIZE} intent")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameter: {total_params:,}")

    print(f"\n[5] Training ({EPOCHS} epoch)...")

    # 6. Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds      = outputs.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

        scheduler.step()
        acc = correct / total

        if epoch % 50 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.4f}")

    print("\n    Training selesai!")

    # 7. Evaluasi akhir
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        outputs  = model(X_tensor)
        loss_val = criterion(outputs, y_tensor).item()
        preds    = outputs.argmax(dim=1)
        acc_val  = (preds == y_tensor).sum().item() / len(y_tensor)

    print(f"\n[6] Evaluasi Akhir:")
    print(f"    Loss    : {loss_val:.4f}")
    print(f"    Accuracy: {acc_val:.4f} ({acc_val*100:.2f}%)")

    # 8. Simpan model & metadata
    model_data = {
        'model_state': model.state_dict(),
        'vocab':       vocab,
        'tags':        tags,
        'input_size':  INPUT_SIZE,
        'hidden1':     HIDDEN1,
        'hidden2':     HIDDEN2,
        'output_size': OUTPUT_SIZE,
        'dropout':     DROPOUT,
    }
    torch.save(model_data, 'chatbot_model.pth')
    print(f"\n    Model disimpan: chatbot_model.pth")
    print("=" * 60)
    print("   TRAINING SELESAI! Jalankan: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    train()
