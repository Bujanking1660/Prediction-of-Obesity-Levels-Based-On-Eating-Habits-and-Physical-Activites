# LAPORAN PRAKTIKUM PEMBELAJARAN MESIN (P2)
## REPRESENTASI DATA, PRA-PEMROSESAN, DAN KLASIFIKASI K-NEAREST NEIGHBORS (KNN)
**Studi Kasus: Klasifikasi Tingkat Obesitas Berdasarkan Kebiasaan Makan dan Kondisi Fisik**

---

### IDENTITAS TUGAS
- **Mata Kuliah:** Pembelajaran Mesin (*Machine Learning*)
- **Topik/Modul:** Tugas P2 — Representasi Data & Pra-pemrosesan Data
- **Algoritma:** *K-Nearest Neighbors* (KNN)
- **Sumber Data:** *Estimation of obesity levels based on eating habits and physical condition* (UCI Machine Learning Repository / Kaggle)
- **File Notebook:** `notebook/KNN_Obesitas_P2.ipynb`
- **File Dataset Mentah:** `data/raw/ObesityDataSet_raw_and_data_sinthetic.csv`
- **File Dataset Terproses:** `data/processed/`

---

## 1. RINGKASAN EKSEKUTIF (ABSTRAK)

Praktikum ini berfokus pada tahapan representasi data dan pra-pemrosesan (*preprocessing*) yang ketat sebelum menerapkan algoritma klasifikasi *K-Nearest Neighbors* (KNN) pada dataset estimasi tingkat obesitas (7 kelas target). Dataset awal berukuran 2.111 baris dan 17 kolom. Melalui proses audit data, ditemukan bahwa sekitar 77% data merupakan data sintetis hasil *SMOTE* yang menyebabkan fitur diskret berskala survei dan usia memiliki nilai pecahan desimal. 

Tahapan pra-pemrosesan yang dilakukan meliputi:
1. Koreksi tipe data dan pembulatan nilai desimal pada fitur diskret.
2. Verifikasi *missing values* (0 data hilang ditemukan).
3. Eliminasi duplikasi data (27 baris duplikat dihapus, menyisakan 2.084 baris).
4. Penyeragaman ejaan dan kapitalisasi nilai kategorikal.
5. Analisis *outlier* dengan metode *Interquartile Range* (IQR) serta *capping* pada variabel usia (`Age`).
6. *Encoding* fitur biner, ordinal, dan nominal (*One-Hot Encoding*).
7. Identifikasi dan penanganan **kebocoran data (*data leakage*)** dengan mengeliminasi fitur tinggi badan (`Height`) dan berat badan (`Weight`) karena keduanya merupakan pembentuk langsung indeks massa tubuh (BMI) yang mendefinisikan label target.

Setelah data dipisahkan secara *stratified* (80% latih, 20% uji), fitur dinormalisasi/distandarisasi menggunakan parameter yang hanya dihitung dari data latih (*training set*). Hasil pengujian model KNN menunjukkan bahwa akurasi data uji terbaik mencapai **69,06%** pada $K=1$. Laporan ini juga menguraikan kelemahan penggunaan $K=1$ akibat keberadaan data sintetis SMOTE serta evaluasi performa per kelas menggunakan metrik *precision*, *recall*, dan *f1-score*.

---

## 2. PENDAHULUAN & LATAR BELAKANG

### 2.1 Masalah
Obesitas merupakan permasalahan kesehatan global yang berisiko memicu berbagai penyakit komorbiditas seperti diabetes mellitus tipe 2, hipertensi, dan penyakit kardiovaskular. Deteksi dini tingkat berat badan seseorang secara non-invasif berdasarkan kebiasaan sehari-hari (pola makan, asupan air, konsumsi kalori tinggi, aktivitas fisik, dan moda transportasi) dapat memberikan rekomendasi preventif sebelum pengukuran klinis mendalam.

### 2.2 Tujuan
1. Melakukan pemahaman data (*data understanding*) serta audit kualitas data mentah.
2. Melakukan pra-pemrosesan data yang tepat: koreksi tipe, penanganan duplikasi, standarisasi kategori, analisis *outlier*, *encoding*, dan penanganan ketidakseimbangan kelas.
3. Mendeteksi dan memitigasi potensi **data leakage** yang dapat mendistorsi validitas model.
4. Menerapkan pembagian data (*train-test split*) dan transformasi fitur (*scaling*) tanpa melanggar prinsip isolasi data uji.
5. Membangun, mengoptimasi nilai hiperparameter $K$, dan mengevaluasi kinerja algoritma *K-Nearest Neighbors* (KNN).

---

## 3. PEMAHAMAN DATA (DATA UNDERSTANDING)

Dataset terdiri dari 2.111 baris dan 17 kolom. Terdapat 16 fitur prediktor dan 1 kolom target bernama asli `NObeyesdad` (yang diubah menjadi `Obesity_Level`).

### Kamus Data
| No | Nama Kolom | Keterangan Variabel | Tipe Data Asli | Tipe Konseptual | Skala Nilai |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | `Age` | Usia responden (tahun) | `float64` | Numerik Rasio | 14 – 61 |
| 2 | `Gender` | Jenis kelamin | `object` | Kategorikal Nominal | Female, Male |
| 3 | `Height` | Tinggi badan (meter) | `float64` | Numerik Rasio | 1.45 – 1.98 |
| 4 | `Weight` | Berat badan (kg) | `float64` | Numerik Rasio | 39.0 – 173.0 |
| 5 | `CALC` | Frekuensi konsumsi alkohol | `object` | Kategorikal Ordinal | no, Sometimes, Frequently, Always |
| 6 | `FAVC` | Sering konsumsi makanan tinggi kalori | `object` | Kategorikal Biner | no, yes |
| 7 | `FCVC` | Frekuensi konsumsi sayur | `float64` | Diskrit Ordinal | 1 – 3 |
| 8 | `NCP` | Jumlah waktu makan utama per hari | `float64` | Diskrit Ordinal | 1 – 4 |
| 9 | `SCC` | Memantau asupan kalori harian | `object` | Kategorikal Biner | no, yes |
| 10 | `SMOKE` | Kebiasaan merokok | `object` | Kategorikal Biner | no, yes |
| 11 | `CH2O` | Konsumsi air minum harian | `float64` | Diskrit Ordinal | 1 – 3 |
| 12 | `family_history_with_overweight` | Riwayat keluarga kelebihan berat badan | `object` | Kategorikal Biner | no, yes |
| 13 | `FAF` | Frekuensi aktivitas fisik mingguan | `float64` | Diskrit Ordinal | 0 – 3 |
| 14 | `TUE` | Waktu penggunaan gawai teknologi | `float64` | Diskrit Ordinal | 0 – 2 |
| 15 | `CAEC` | Frekuensi makan camilan di sela makan | `object` | Kategorikal Ordinal | no, Sometimes, Frequently, Always |
| 16 | `MTRANS` | Moda transportasi utama | `object` | Kategorikal Nominal | 5 kategori transportasi |
| 17 | `NObeyesdad` | **Target:** Tingkat klasifikasi obesitas | `object` | Kategorikal Multikelas | 7 kelas |

---

## 4. TAHAP PRA-PEMROSESAN DATA (DATA PREPROCESSING)

### 4.1 Koreksi Tipe Kolom & Pembulatan Desimal
- **Temuan:** Kolom survei (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`) serta `Age` memiliki banyak nilai desimal tak lazim:
  - `Age`: 1.375 baris bernilai pecahan.
  - `CH2O`: 1.290 baris bernilai pecahan.
  - `FAF`: 1.208 baris bernilai pecahan.
  - `TUE`: 1.153 baris bernilai pecahan.
  - `FCVC`: 826 baris bernilai pecahan.
  - `NCP`: 640 baris bernilai pecahan.
- **Penyebab:** Sekitar 77% data dibuat secara sintetis menggunakan teknik *SMOTE (Synthetic Minority Over-sampling Technique)* yang melakukan interpolasi linier kontinu antar tetangga terdekat.
- **Tindakan:** Dilakukan pembulatan matematika (`round().astype(int)`) agar data kembali ke skala aslinya (diskret/skor bulat).

### 4.2 Pemeriksaan Missing Values
- Dilakukan pengecekan terhadap nilai `NaN`/`null` serta nilai teks tersembunyi (seperti spasi kosong, string `?`, `None`, `-`).
- **Hasil:** Ditemukan 0 nilai hilang (data lengkap 100%).

### 4.3 Penanganan Duplikasi
- **Temuan:** Setelah proses pembulatan diskret, terdeteksi **27 baris duplikat persis** (1,28% dari total data). Pada file mentah awal terdapat 24 baris duplikat.
- **Tindakan:** Seluruh 27 baris duplikat dihapus (*drop duplicates*) sebelum data dibagi menjadi data latih dan data uji.
- **Justifikasi:** Jika duplikasi tidak dibuang sebelum *split*, rekaman yang identik berpotensi muncul di kedua partisi (*train* dan *test*), menciptakan ilusi akurasi buatan (*data leakage*).
- **Hasil:** Ukuran data berkurang dari 2.111 menjadi **2.084 baris**.

### 4.4 Penyeragaman Kategori & Standarisasi Teks
- Dilakukan penghapusan spasi di awal/akhir (*strip*) pada semua atribut bertipe string.
- Dilakukan penyeragaman kapitalisasi (*capitalized*): mengubah entri inkonsisten seperti `'no'` dan `'yes'` menjadi `'No'` dan `'Yes'`, serta merapikan nilai ordinal `'Sometimes'`, `'Frequently'`, dan `'Always'`.

### 4.5 Analisis Outlier (Metode Interquartile Range / IQR)
Dilakukan evaluasi pencilan pada fitur numerik kontinu:

$$\text{IQR} = Q_3 - Q_1$$
$$\text{Batas Bawah} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Batas Atas} = Q_3 + 1.5 \times \text{IQR}$$

**Tabel Hasil Analisis IQR:**
| Fitur | Batas Bawah | Batas Atas | Jumlah Outlier | Persentase | Tindakan |
|---|:---:|:---:|:---:|:---:|---|
| `Age` | 11.00 tahun | 35.00 tahun | 160 | 7,68% | **Capping** (menggunakan batas data latih) |
| `Height` | 1,42 m | 1,98 m | 1 | 0,05% | Dipertahankan (masuk akal biologis) |
| `Weight` | 2,96 kg | 171,06 kg | 1 | 0,05% | Dipertahankan (masuk akal biologis) |

*Catatan:* Outlier usia tidak dihapus agar variasi usia dewasa tidak hilang, melainkan dilakukan *winsorizing/capping* (dibatasi maksimal 35 tahun) berdasarkan batas yang dihitung murni dari data latih.

### 4.6 Analisis Ketidakseimbangan Kelas (Class Imbalance)
Distribusi frekuensi 7 kelas target pada data bersih (2.084 baris):
1. `Insufficient_Weight` : 266 baris (12,76%)
2. `Normal_Weight` : 282 baris (13,53%)
3. `Overweight_Level_I` : 276 baris (13,24%)
4. `Overweight_Level_II` : 288 baris (13,82%)
5. `Obesity_Type_I` : 351 baris (16,84%)
6. `Obesity_Type_II` : 297 baris (14,25%)
7. `Obesity_Type_III` : 324 baris (15,55%)

**Rasio kelas terbesar terhadap terkecil adalah 1,32 : 1**. Data tergolong sangat seimbang (*balanced*), yang merupakan konsekuensi langsung dari penggunaan SMOTE oleh pembuat dataset. Oleh karena itu, teknik resampling tambahan tidak diperlukan, namun pembagian data wajib menerapkan pembagian terstratifikasi (*stratified sampling*).

### 4.7 Encoding Fitur Kategorikal
Transformasi numerik dilakukan berdasarkan skala pengukuran:
1. **Fitur Biner (0 atau 1):** `Gender` (Female=0, Male=1), `FAVC`, `SCC`, `SMOKE`, `family_history_with_overweight` (No=0, Yes=1).
2. **Fitur Ordinal (Integer berurutan):** `CALC` dan `CAEC` dipetakan ke tingkat konsumsi: `No`: 0, `Sometimes`: 1, `Frequently`: 2, `Always`: 3.
3. **Fitur Nominal:** `MTRANS` diubah menggunakan *One-Hot Encoding* menghasilkan 5 kolom dummy biner: `MTRANS_Automobile`, `MTRANS_Bike`, `MTRANS_Motorbike`, `MTRANS_Public_Transportation`, `MTRANS_Walking`.
4. **Target:** `Obesity_Level` dipetakan ke bilangan bulat 0 sampai 6.

Total dimensi data setelah *encoding* bertambah menjadi 21 kolom (termasuk target).

---

## 5. ANALISIS MENDALAM: KEBOCORAN DATA (DATA LEAKAGE)

### 5.1 Bukti Kebocoran Rumus BMI
Label target `Obesity_Level` secara klinis ditentukan oleh rumus Indeks Massa Tubuh (*Body Mass Index* / BMI):

$$\text{BMI} = \frac{\text{Weight (kg)}}{\left(\text{Height (m)}\right)^2}$$

Ketika nilai BMI dihitung pada dataset, rentang nilai per kelas memperlihatkan batas deterministik yang hampir sempurna:
- `Insufficient_Weight`: BMI 13,0 – 19,1 (rata-rata: 17,4)
- `Normal_Weight`: BMI 18,5 – 24,9 (rata-rata: 22,0)
- `Overweight_Level_I`: BMI 22,8 – 28,8 (rata-rata: 26,0)
- `Overweight_Level_II`: BMI 25,7 – 30,4 (rata-rata: 28,2)
- `Obesity_Type_I`: BMI 29,9 – 35,2 (rata-rata: 32,3)
- `Obesity_Type_II`: BMI 34,0 – 39,8 (rata-rata: 36,7)
- `Obesity_Type_III`: BMI 36,8 – 50,8 (rata-rata: 42,3)

### 5.2 Pengujian Empiris Leakage (5-Fold Cross Validation)
Untuk membuktikan dampak kebocoran informasi ini, dilakukan perbandingan eksperimen menggunakan 5-Fold Stratified Cross Validation pada data latih:
- **Skenario A (Dengan Weight & Height):** Akurasi rata-rata = **77,83%** (Std: 0,0143)
- **Skenario B (Tanpa Weight & Height):** Akurasi rata-rata = **72,55%** (Std: 0,0123)

### 5.3 Keputusan Ilmiah
Fitur `Weight` dan `Height` **wajib dieliminasi dari himpunan fitur akhir**. 
*Alasan:* Tujuan esensial dari proyek pembelajaran mesin ini adalah memprediksi kecenderungan obesitas **berdasarkan pola hidup dan perilaku kebiasaan makan**. Jika berat dan tinggi badan disertakan, model tidak belajar pola kebiasaan hidup, melainkan sekadar merekonstruksi rumus aproksimasi BMI. Setelah kedua kolom dibuang, dataset final memiliki **18 fitur prediktor**.

---

## 6. PEMBAGIAN DATA DAN TRANSFORMASI SKALASI (SCALING)

### 6.1 Stratified Train-Test Split
Dataset 2.084 baris dibagi dengan rasio **80% data latih (1.667 baris)** dan **20% data uji (417 baris)** dengan parameter `random_state=42` dan `stratify=y`.
Proporsi kelas pada kedua partisi terdistribusi identik (masing-masing kelas berkisar antara 12,7% – 16,9%).

### 6.2 Normalisasi vs Standarisasi
Karena algoritma KNN menghitung jarak Euclidean di ruang $N$-dimensi:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

Atribut dengan rentang nilai besar (seperti usia 14–35) akan mendominasi atribut biner (0–1).
Untuk mencegah kebocoran data (*data leakage*), transformator skalasi hanya di-*fit* pada data latih (`X_train`), kemudian diterapkan (*transform*) pada data latih dan data uji (`X_test`).

**Perbandingan Metode Scaling (5-Fold CV pada Data Latih, K=5):**
| Metode Transformasi | Akurasi Rata-rata CV | Standar Deviasi |
|---|:---:|:---:|
| Tanpa Skalasi | 71,09% | 0,0145 |
| Normalisasi (Min-Max Scaler) | 69,05% | 0,0208 |
| Standarisasi (Z-score / StandardScaler) | 68,57% | 0,0104 |

*Catatan:* Standarisasi Z-score dipilih untuk pipeline model final karena menghasilkan standar deviasi paling stabil serta lebih tahan terhadap distribusi fitur ordinal.

---

## 7. PEMODELAN K-NEAREST NEIGHBORS (KNN) & EVALUASI

### 7.1 Pencarian Nilai K Optimal (Hyperparameter Tuning)
Pengujian nilai $K$ dilakukan dari rentang $K = 1$ hingga $K = 30$ menggunakan *Stratified 5-Fold Cross Validation* pada data latih (data uji belum disentuh).
- **Hasil:** Akurasi CV tertinggi tercapai pada **$K = 1$** dengan akurasi **70,07%**.
- Seiring bertambahnya $K$, akurasi mengalami tren penurunan perlahan ($K=5$: ~68,6%, $K=15$: ~66%, $K=30$: ~63%).

### 7.2 Analisis Kritis terhadap Pilihan $K=1$
Secara teori, $K=1$ memiliki batas keputusan (*decision boundary*) yang sangat kompleks, sangat sensitif terhadap *noise*, dan rentan mengalami *overfitting*.
Namun, pada dataset ini $K=1$ mencatat akurasi tertinggi karena **77% data merupakan hasil interpolasi SMOTE**. Titik-titik sintetis SMOTE menghasilkan pasangan tetangga yang sangat dekat di ruang fitur. Jika satu titik berada di *train fold* dan 'kembaran' sintetisnya berada di *validation fold*, $K=1$ dengan mudah memprediksi label yang benar hanya dengan menyalin label tetangga terdekatnya. Ini adalah bentuk *leakage terselubung* bawaan dari struktur dataset SMOTE.

### 7.3 Hasil Evaluasi Akhir pada Data Uji (Test Set)
Model KNN final ($K=1$, *StandardScaler*) dilatih pada seluruh 1.667 data latih dan diuji **satu kali** pada 417 data uji yang independen.

**Akurasi Keseluruhan Data Uji:** **69,06% (~69,1%)**

**Laporan Klasifikasi Komprehensif (Classification Report):**
| Kelas Tingkat Obesitas | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `Insufficient_Weight` | 0,700 | 0,792 | 0,743 | 53 |
| `Normal_Weight` | 0,500 | 0,339 | 0,404 | 56 |
| `Overweight_Level_I` | 0,571 | 0,655 | 0,610 | 55 |
| `Overweight_Level_II` | 0,612 | 0,517 | 0,561 | 58 |
| `Obesity_Type_I` | 0,618 | 0,671 | 0,644 | 70 |
| `Obesity_Type_II` | 0,836 | 0,850 | 0,843 | 60 |
| `Obesity_Type_III` | 0,900 | 0,969 | 0,933 | 65 |
| **Akurasi Rata-rata (Accuracy)** | | | **0,691** | **417** |
| **Macro Average** | **0,677** | **0,685** | **0,677** | **417** |
| **Weighted Average** | **0,681** | **0,691** | **0,682** | **417** |

### 7.4 Analisis Kinerja per Kelas & Matriks Konfusi (Confusion Matrix)
1. **Performa Sangat Tinggi pada Kelas Ekstrem:**
   - `Obesity_Type_III` memperoleh **F1-Score 0,933** (Recall 96,9%, Precision 90,0%).
   - `Obesity_Type_II` memperoleh **F1-Score 0,843** (Recall 85,0%, Precision 83,6%).
   - *Penyebab:* Individu dengan obesitas tingkat tinggi memiliki profil kebiasaan makan dan fisik yang sangat khas dan terpisah jauh di ruang fitur.
2. **Kinerja Rendah pada Kelas Tengah (Perbatasan):**
   - `Normal_Weight` memperoleh kinerja terendah (**F1-Score 0,404**, Recall hanya 33,9%).
   - `Overweight_Level_I` & `Overweight_Level_II` memiliki F1-score di kisaran 0,56 – 0,61.
   - *Penyebab:* Terdapat tumpang tindih (*overlap*) kebiasaan makan antara individu dengan berat badan normal dan individu kelebihan berat badan tingkat awal. Seseorang dengan berat normal dan kelebihan berat badan sering kali memiliki gaya hidup yang mirip, sehingga jarak fiturnya sangat berdekatan.

---

## 8. RINGKASAN METRIK PRAKTIKUM

| Parameter / Metrik | Nilai / Keterangan |
|---|---|
| Ukuran data awal mentah | 2.111 baris, 17 kolom |
| Jumlah nilai hilang (*Missing values*) | 0 nilai |
| Duplikasi data terhapus | 27 baris (1,28%) |
| Ukuran data bersih | 2.084 baris |
| Fitur yang dieliminasi karena *leakage* | 2 fitur (`Weight`, `Height`) |
| Jumlah fitur prediktor final | 18 fitur |
| Rasio ketidakseimbangan kelas awal | 1,32 : 1 (seimbang akibat SMOTE) |
| Akurasi CV KNN dengan *leakage* ($K=5$) | **77,83%** |
| Akurasi CV KNN tanpa *leakage* ($K=5$) | **72,55%** |
| Nilai $K$ terbaik hasil tuning CV | **$K = 1$** (Akurasi CV: 70,07%) |
| **Akurasi akhir pada data uji (*Test Set*)** | **69,06%** |
| Macro F1-Score pada data uji | **67,70%** |
| Weighted F1-Score pada data uji | **68,20%** |

---

## 9. KESIMPULAN & SARAN

### 9.1 Kesimpulan
1. **Pentingnya Audit Kualitas Data:** Mengidentifikasi bahwa dataset didominasi sampel sintetis SMOTE (77%) sangat krusial untuk memahami anomali nilai pecahan pada fitur survei dan usia.
2. **Eliminasi Data Leakage adalah Wajib:** Menyertakan berat dan tinggi badan meningkatkan akurasi secara semu (77,83%), namun merusak tujuan pemodelan prediktif berbasis gaya hidup. Mengeliminasi kedua fitur tersebut mengembalikan integritas model dengan akurasi riil 69,06%.
3. **Kepatuhan Protokol Evaluasi:** Menghapus duplikasi sebelum *split*, menerapkan *capping* outlier dan *scaling* hanya berdasarkan statistik data latih berhasil mencegah kebocoran informasi dari data uji.
4. **Karakteristik KNN:** KNN bekerja sangat efektif pada kelas-kelas obesitas ekstrem (`Obesity Type II` dan `III`), namun kesulitan membedakan batas halus antara kelas `Normal Weight` dan `Overweight Level I/II`.

### 9.2 Saran Pengembangan Lanjutan
1. **Eksplorasi Algoritma Non-Jarak:** Mencoba model berbasis pohon keputusan (*Decision Tree*, *Random Forest*, atau *XGBoost*) yang lebih tangguh terhadap data campuran (kategorikal & numerik) tanpa ketergantungan jarak metrik Euclidean.
2. **Feature Engineering:** Membuat fitur interaksi baru antara aktivitas fisik (`FAF`), waktu santai gawai (`TUE`), dan konsumsi sayur (`FCVC`) untuk mempertegas pemisah kelas menengah.
3. **Pengujian pada Data Riil Murni:** Menguji model pada subset 23% data asli (non-SMOTE) untuk mengukur kemampuan generalisasi model pada populasi dunia nyata.
