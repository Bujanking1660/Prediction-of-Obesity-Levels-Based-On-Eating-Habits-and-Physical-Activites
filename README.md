# Prediction of Obesity Levels Based On Eating Habits and Physical Activities
**Tugas Praktikum Pembelajaran Mesin (P2)** — Representasi Data, Pra-pemrosesan Data, & Klasifikasi *K-Nearest Neighbors* (KNN).

Repository ini berisi eksperimen pra-pemrosesan data menyeluruh, analisis *data leakage*, serta pemodelan klasifikasi tingkat obesitas (7 kelas) berdasarkan pola makan dan aktivitas fisik responden menggunakan algoritma *K-Nearest Neighbors* (KNN).

---

## 📁 Struktur Direktori (*Directory Tree*)

```text
Prediction-of-Obesity-Levels-Based-On-Eating-Habits-and-Physical-Activites/
│
├── data/
│   ├── raw/
│   │   └── ObesityDataSet_raw_and_data_sinthetic.csv   # Dataset mentah dari UCI / Kaggle (2.111 baris, 17 kolom)
│   │
│   └── processed/
│       ├── obesity_cleaned.csv                         # Data bersih (2.084 baris): koreksi desimal, drop duplikat, standardisasi teks
│       ├── obesity_encoded.csv                         # Data hasil encoding biner, ordinal, & one-hot encoding (2.084 baris, 21 kolom)
│       ├── train_preprocessed.csv                      # Data latih final (1.667 baris, 19 kolom): bebas leakage, capped, Z-score scaled
│       └── test_preprocessed.csv                       # Data uji final (417 baris, 19 kolom): ditransformasi konsisten dari data train
│
├── doc/
│   └── Laporan_KNN_Obesitas_P2.md                      # Laporan komprehensif praktikum format akademik (analisis bab per bab)
│
├── notebook/
│   └── KNN_Obesitas_P2.ipynb                           # Jupyter Notebook alur lengkap pra-pemrosesan, EDA, tuning K, & evaluasi KNN
│
├── output/
│   ├── knn_model.joblib                                # Model KNN terlatih (K=1) yang telah diekspor (artefak biner)
│   └── confusion_matrix.png                            # Plot visualisasi matriks konfusi evaluasi model pada data uji
│
└── README.md                                           # Dokumentasi struktur proyek & panduan penggunaan
```

---

## 📊 Penjelasan Data Hasil Pra-pemrosesan (*Preprocessed Data*)

Seluruh data hasil tahapan pra-pemrosesan disimpan di folder `data/processed/`:

1. **`obesity_cleaned.csv`**
   - **Ukuran:** 2.084 baris × 17 kolom
   - **Tahapan:**
     - Mengoreksi nilai desimal tak wajar pada kolom skala survei diskret (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) dan usia (`Age`) menjadi bilangan bulat (`int`). Nilai desimal ini merupakan artefak interpolasi SMOTE pada data sintetis (~77% data).
     - Menghapus 27 baris data duplikat persis untuk mencegah potensi kebocoran data (*data leakage*).
     - Membersihkan spasi kosong (*whitespace*) dan menyeragamkan kapitalisasi teks kategori (`No`/`Yes`, `Sometimes`, `Frequently`, `Always`).
     - Mengganti nama kolom target menjadi `Obesity_Level`.
     - Masih mempertahankan label kategori asli dalam bentuk teks sehingga siap digunakan untuk visualisasi eksplorasi (EDA).

2. **`obesity_encoded.csv`**
   - **Ukuran:** 2.084 baris × 21 kolom
   - **Tahapan:**
     - *Binary Encoding* (0/1): `Gender`, `FAVC`, `SCC`, `SMOKE`, `family_history_with_overweight`.
     - *Ordinal Encoding* (0 s.d. 3): `CALC`, `CAEC`.
     - *One-Hot Encoding*: `MTRANS` ditransformasikan menjadi 5 kolom biner dummy (`MTRANS_Automobile`, `MTRANS_Bike`, `MTRANS_Motorbike`, `MTRANS_Public_Transportation`, `MTRANS_Walking`).
     - *Target Mapping*: `Obesity_Level` dipetakan ke integer 0 sampai 6.

3. **`train_preprocessed.csv` & `test_preprocessed.csv`**
   - **Ukuran Data Latih:** 1.667 baris × 19 kolom (80% split terstratifikasi)
   - **Ukuran Data Uji:** 417 baris × 19 kolom (20% split terstratifikasi)
   - **Tahapan Kritis:**
     - **Eliminasi Kebocoran Data (*Data Leakage*):** Kolom `Weight` dan `Height` **dibuang** karena merupakan pembentuk langsung label BMI. Tujuannya agar model benar-benar memprediksi tingkat obesitas dari **kebiasaan hidup dan pola makan**.
     - **Penanganan Outlier:** Fitur `Age` di-*capping* (*winsorizing*) batas atas IQR 35 tahun berdasarkan statistik data latih saja.
     - **Feature Scaling:** Seluruh fitur numerik distandarisasi (*StandardScaler / Z-score*) dengan parameter rata-rata dan deviasi standar yang di-*fit* murni pada data latih untuk menjaga integritas data uji.

---

## 🚀 Cara Menjalankan (*Quickstart*)

### 1. Prasyarat (*Dependencies*)
Pastikan Python 3.8+ terpasang beserta pustaka berikut:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 2. Menjalankan Notebook
Buka Jupyter Notebook di direktori `notebook/`:
```bash
cd notebook
jupyter notebook KNN_Obesitas_P2.ipynb
```
*Catatan:* Notebook sudah dikonfigurasi untuk mendeteksi lokasi dataset secara otomatis baik saat dijalankan dari folder `notebook/` maupun dari *root directory*.

---

## 📈 Ringkasan Temuan & Hasil Model

| Parameter / Metrik | Nilai / Keterangan |
|---|---|
| Algoritma | *K-Nearest Neighbors* (KNN) |
| Jumlah Fitur Prediktor Final | 18 fitur (tanpa `Weight` & `Height`) |
| Pengaruh Skalasi (CV K=5) | Z-score scaling menghasilkan variansi terendah (std: 0,0104) |
| Nilai K Optimal (5-Fold CV) | $K = 1$ (Akurasi CV: 70,07%) |
| **Akurasi Data Uji (*Test Set*)** | **69,06% (~69,1%)** |
| Macro F1-Score | 67,7% |
| Weighted F1-Score | 68,2% |

Detail analisis teoritis, matriks konfusi, dan pembahasan mendalam mengenai bahaya interpretasi $K=1$ pada data sintetis SMOTE dapat dibaca pada [Laporan_KNN_Obesitas_P2.md](doc/Laporan_KNN_Obesitas_P2.md).
