# Klasifikasi Teks Promosi Judi Online Berbahasa Indonesia pada Media Sosial dan Komentar YouTube Menggunakan Machine Learning dan Deep Learning

## Identitas Kelompok

| Nama | NIM | Program Studi |
| :--- | :--- | :--- |
| Alan Herva Ikhsan Saputra | 244311001 | Teknologi Rekayasa Perangkat Lunak |
| Davin Rafael Suryatmoko | 244311007 | Teknologi Rekayasa Perangkat Lunak |

- **Mata Kuliah:** Data Science
- **Dosen Pengampu:** Gus Nanang Syaifuddiin, S.Kom., M.Kom.
- **Dokumentasi Lengkap:** Lihat [Laporan Proyek ML (L-M-O-P)](docs/Laporan_Proyek_ML_LMOP.md) dan [Formulir Pengajuan Ide](docs/09_LAMPIRAN_B.md)

---

## Deskripsi Proyek

Proyek Data Science ini dirancang untuk mendeteksi dan mengklasifikasikan teks promosi judi online berbahasa Indonesia secara otomatis ke dalam dua kelas (*Binary Classification*), yaitu **Promosi Judi Online (`1`)** dan **Bukan Promosi (`0`)**. Penyebaran promosi judi online di media sosial dan kolom komentar YouTube kerap menggunakan variasi penulisan tidak baku (seperti *leetspeak* dan karakter simbolik Unicode) untuk menghindari filter otomatis. Melalui komparasi pendekatan *Machine Learning* dan *Deep Learning*, proyek ini bertujuan menyediakan solusi penyaringan konten yang akurat dan efisien.

---

## Dataset

Penelitian ini menggabungkan dua sumber dataset teks berbahasa Indonesia dari empat platform digital (**YouTube, X, TikTok, dan Facebook**):

1. **Indonesian Online Gambling Promotion Multimodal Dataset** — [Hugging Face (`0xRafie`)](https://huggingface.co/datasets/0xRafie/indonesian-gambling-multimodal-dataset) (5.028 data dari X, TikTok, dan Facebook; penelitian ini hanya memanfaatkan fitur teks `text_caption`).
2. **YouTube Comments Gambling Dataset** — hasil *crawling* mandiri dari kolom komentar YouTube berbahasa Indonesia (6.219 data).

**Ringkasan Statistik Dataset:**
- **Total Data Mentah:** 11.247 baris (6.381 bukan promosi, 4.866 promosi judi online)
- **Total Data Bersih (setelah *cleaning* & deduplikasi):** 10.558 baris unik (6.098 bukan promosi [57,76%], 4.460 promosi [42,24%])
- **Pembagian Data (*Multi-Stratified Split* 80 : 10 : 10):**
  - **Training Set:** 8.446 sampel
  - **Validation Set:** 1.056 sampel
  - **Test Set:** 1.056 sampel

---

## Metode

Pengembangan sistem mengikuti kerangka **ML Lifecycle (L-M-O-P)** dengan membandingkan tiga tingkat kompleksitas model klasifikasi teks:

1. **Preprocessing & Ekstraksi Fitur:**
   - Pembersihan teks meliputi normalisasi Unicode (`NFKC`), *case folding*, penghapusan URL/`@mention`/simbol, serta normalisasi *leetspeak* (seperti `g4cor` $\rightarrow$ `gacor`).
   - **TF-IDF Vectorizer** (`max_features=5000`, unigram & bigram) untuk model *Machine Learning* klasik.
   - **Tokenizer & Sequence Padding** (`num_words=10000`, `maxlen=100`) untuk model *Deep Learning*.
2. **Model 1 – Baseline Model: Multinomial Naive Bayes**  
   Model probabilistik berbasis kemunculan kata (*bag-of-words*) sebagai tolok ukur performa awal (*lower-bound benchmark*) yang cepat dan ringan untuk klasifikasi teks.
3. **Model 2 – Advanced ML Model: Random Forest**  
   Model *ensemble* berbasis gabungan banyak *Decision Tree* untuk menangkap interaksi fitur non-linear antar kata/frasa pada matriks TF-IDF.
4. **Model 3 – Deep Learning Model: Long Short-Term Memory (LSTM)**  
   Arsitektur *Recurrent Neural Network* (RNN) dengan layer *Embedding* untuk mempelajari konteks dan ketergantungan urutan kata dalam kalimat secara sekuensial.

**Metrik Evaluasi:**
- **F1-Score** (metrik utama), **Accuracy**, **Precision**, dan **Recall** untuk mengukur keseimbangan ketepatan deteksi konten promosi maupun non-promosi.
- **Confusion Matrix**, kurva *Training vs Validation Loss/Accuracy*, serta **Training & Inference Time**.

---

## Tujuan & Keterkaitan SDGs

- Mengolah dan membersihkan gabungan dataset teks promosi judi online dari media sosial dan komentar YouTube agar siap digunakan tanpa *data leakage*.
- Membangun, mengoptimasi, dan membandingkan performa **Multinomial Naive Bayes**, **Random Forest**, dan **LSTM** dengan target capaian **F1-Score > 0,85** pada *test set*.
- Mendukung pencapaian **SDG 3 – Good Health and Well-Being (Target 3.4)** melalui pengembangan model deteksi otomatis yang membantu mengurangi paparan konten promosi judi online di ruang digital demi menjaga kesehatan mental dan kesejahteraan masyarakat.

> **Catatan Etika:** Dataset yang digunakan hanya memuat teks publik untuk keperluan penelitian akademik tanpa mengumpulkan atau mengidentifikasi informasi pribadi pemilik akun.

---

## Struktur Repository

```text
uas-Klasifikasi-Promosi-Judi-Online/
├── data/
│   ├── raw/                                # Dataset mentah (hf_dataset.csv & youtube_comments_dataset.csv)
│   └── processed/                          # Hasil split bersih (train.csv, val.csv, test.csv)
├── docs/
│   ├── 09_LAMPIRAN_B.md                    # Formulir pengajuan ide (Checkpoint 1)
│   └── Laporan_Proyek_ML_LMOP.md           # Laporan lengkap mengikuti kerangka ML Lifecycle (L-M-O-P)
├── models/                                 # Vectorizer, tokenizer, dan model terlatih (.joblib / .keras)
├── notebook/
│   ├── 01_learning_problem_eda.ipynb       # [L] Data Understanding & Visualisasi EDA
│   ├── 02_data_preparation.ipynb           # [L] Data Cleaning, Splitting, & Feature Extraction
├── reports/
│   └── figures/                            # Grafik hasil EDA, training history, dan confusion matrix
└── README.md
