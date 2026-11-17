## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen | Isian Mahasiswa |
|----------|-----------------|
| Nama / NIM | Alan Herva Ikhsan Saputra (244311001) dan Davin Rafael Suryatmoko (244311007) |
| Judul ide | Klasifikasi Teks Promosi Judi Online Berbahasa Indonesia pada Media Sosial dan Komentar YouTube Menggunakan Machine Learning dan Deep Learning |
| SDG & target yang dipilih | SDG 3 – Good Health and Well-Being (Kehidupan Sehat dan Sejahtera), Target 3.4. Proyek ini diharapkan dapat mendukung upaya mengurangi paparan konten promosi judi online yang berpotensi berdampak buruk terhadap kesehatan mental dan kesejahteraan masyarakat. |
| Masalah nyata yang diangkat | Promosi judi online masih banyak ditemukan di media sosial dan kolom komentar YouTube dengan berbagai variasi penulisan kalimat. Banyaknya konten tersebut membuat proses identifikasi secara manual membutuhkan waktu dan tenaga. Oleh karena itu, diperlukan model Machine Learning dan Deep Learning yang dapat membantu mengklasifikasikan teks promosi judi online dan bukan promosi secara otomatis. |
| Tugas ML (Target & jenis tugas) | Target: Kolom `label`, dengan nilai `1` = promosi judi online dan `0` = bukan promosi judi online. Jenis tugas: *Supervised Learning – Binary Classification* (klasifikasi dua kelas). |
| Objective & metrik utama | Meminimalkan kesalahan prediksi menggunakan fungsi loss *Cross-Entropy* dan membandingkan performa tiga model klasifikasi teks. Metrik utama: F1-Score. Metrik pendukung: Accuracy, Precision, Recall, dan Confusion Matrix. |
| Dataset (sumber, URL, ukuran, jenis data) | 1. *Indonesian Online Gambling Promotion Multimodal Dataset* (Sumber: Hugging Face – `0xRafie`, URL: https://huggingface.co/datasets/0xRafie/indonesian-gambling-multimodal-dataset, Ukuran: 5.028 data dari platform Facebook, TikTok, dan X, Jenis data: Teks berbahasa Indonesia dan gambar). <br> 2. *YouTube Comments Gambling Dataset* (Sumber: Hasil *crawling* mandiri komentar YouTube berbahasa Indonesia, Ukuran: 6.219 data, Jenis data: Teks berbahasa Indonesia). <br> Total keseluruhan setelah digabung berjumlah 11.247 data berlabel biner. |
| Rencana tiga model (baseline / advanced / deep learning) | 1. Baseline: Multinomial Naive Bayes. <br> 2. Advanced: Random Forest. <br> 3. Deep Learning: Long Short-Term Memory (LSTM). <br> Ketiga model akan dilatih dan dibandingkan menggunakan data pengujian (*test set*) dan metrik evaluasi yang sama. |
| Risiko & pertimbangan etika (bias, privasi) | Dataset memiliki kemungkinan bias karena berasal dari beberapa platform media sosial dan kolom komentar YouTube dengan gaya bahasa yang beragam (singkatan, bahasa tidak baku, simbol, dan karakter Unicode). Untuk menjaga privasi, penelitian hanya menggunakan teks konten untuk kepentingan akademik dan tidak mengumpulkan maupun mengidentifikasi identitas pemilik akun. |
| Referensi awal (minimal 2) | 1. Riza, F. J. M., Guntara, R. G., & Nugraha, M. R. (2025). Implementasi Algoritma Naive Bayes untuk Filtrasi Spam Komentar Judi Online pada YouTube. *Indonesian Journal of Digital Business*, 5(2), 411–423. https://doi.org/10.17509/ijdb.v5i2.88641 <br> 2. Pratama, M. L. A., Yusuf, M. E., Ramdhani, M. R., Fathoni, & Ibrahim, A. (2026). Deteksi dan Klasifikasi Konten Promosi Judi Online pada Media Sosial YouTube Menggunakan Pendekatan Natural Language Processing dan Machine Learning. *JATI (Jurnal Mahasiswa Teknik Informatika)*, 10(3). https://doi.org/10.36040/jati.v10i3.18440 |


**Kriteria persetujuan dosen:**
- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
