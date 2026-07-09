================================================================
  CHATBOT AKADEMIK ITI — TEKNIK INFORMATIKA
  Implementasi ANN + NLP dengan PyTorch & Flask
================================================================

STRUKTUR FOLDER:
  chatbot_iti/
  ├── dataset.json          ← Dataset intent & respons
  ├── train.py              ← Training model ANN
  ├── app.py                ← Web server Flask
  ├── requirements.txt      ← Library yang dibutuhkan
  ├── chatbot_model.pth     ← Model tersimpan (setelah training)
  └── templates/
      └── index.html        ← Tampilan web chatbot

================================================================
CARA MENJALANKAN:
================================================================

LANGKAH 1 — Install library:
  pip install torch flask numpy

LANGKAH 2 — Training model (lakukan SEKALI):
  cd chatbot_iti
  python train.py

  Tunggu sampai muncul: "TRAINING SELESAI!"
  File chatbot_model.pth akan terbuat otomatis.

LANGKAH 3 — Jalankan web app:
  python app.py

LANGKAH 4 — Buka browser:
  http://localhost:5000

================================================================
DAFTAR INTENT (12 topik):
================================================================
  1. salam           - Sapaan & pembuka percakapan
  2. perpisahan      - Penutup percakapan
  3. krs             - Informasi pengisian KRS
  4. jadwal_kuliah   - Jadwal perkuliahan
  5. nilai           - Cek nilai & IPK
  6. tugas_akhir     - Prosedur & syarat TA/Skripsi
  7. lokasi          - Lokasi kampus & fasilitas
  8. pembayaran      - Biaya kuliah & beasiswa
  9. dosen           - Info & kontak dosen
  10. cuti           - Prosedur cuti akademik
  11. wisuda         - Syarat & prosedur wisuda
  12. kemampuan_bot  - Info tentang kemampuan chatbot

================================================================
ARSITEKTUR MODEL ANN:
================================================================
  Input Layer   → Bag of Words (vocab size)
  Hidden Layer 1 → 128 neuron + BatchNorm + ReLU + Dropout(0.3)
  Hidden Layer 2 → 64 neuron  + BatchNorm + ReLU + Dropout(0.3)
  Output Layer  → 12 neuron (jumlah intent) + Softmax

  Optimizer : Adam (lr=0.001, weight_decay=1e-4)
  Loss      : CrossEntropyLoss
  Epochs    : 300
  Batch Size: 8

================================================================
PIPELINE NLP:
================================================================
  1. Case Folding  → ubah ke huruf kecil
  2. Cleaning      → hapus karakter non-alfabet
  3. Tokenizing    → pisah menjadi kata-kata
  4. Stopword Rem  → hapus kata umum (yang, dan, di, ...)
  5. Stemming      → potong imbuhan (me-, di-, -kan, -an, ...)
  6. Bag of Words  → representasi vektor biner

================================================================
