# Supervised Learning: Classification dengan K-Nearest Neighbors (KNN)

## 📋 Deskripsi Program

Program ini merupakan implementasi lengkap dari algoritma **K-Nearest Neighbors (KNN)** untuk klasifikasi data menggunakan dataset **Iris**. Program mendemonstrasikan seluruh tahapan machine learning dari exploratory data analysis, training, hyperparameter tuning, hingga evaluasi model dan visualisasi prediksi.

---

## 📊 Dataset

**Iris Dataset**
- **Sumber**: `Iris.csv`
- **Jumlah Sampel**: 150 data (105 training, 45 testing dengan test_size=0.30)
- **Fitur/Features**:
  - `SepalLengthCm`: Panjang sepal (cm)
  - `SepalWidthCm`: Lebar sepal (cm)
  - `PetalLengthCm`: Panjang petal (cm)
  - `PetalWidthCm`: Lebar petal (cm)
  
- **Target/Label** (Species - 3 kelas):
  - 0 = Iris-setosa
  - 1 = Iris-versicolor
  - 2 = Iris-virginica

---

## 🔍 Tahapan Program

### 1. **Preparation (Persiapan)**
```python
Import library: pandas, numpy, matplotlib, seaborn, sklearn
Load dataset Iris → Drop kolom ID → Encode Species dengan LabelEncoder
```

### 2. **Exploratory Data Analysis (EDA)**
Analisis mendalam terhadap dataset:
- ✅ Info dataset (tipe data, missing values)
- ✅ Statistik deskriptif (mean, std, min, max, dll)
- ✅ Distribusi jenis bunga iris dengan countplot
- ✅ Boxplot fitur numerik per species
- ✅ Heatmap korelasi antar fitur
- ✅ Scatter plot untuk visualisasi hubungan antar fitur
- ✅ Histogram distribusi semua fitur

### 3. **Feature Engineering**
- Menyiapkan data untuk training
- Copy dataset ke variabel `train`

### 4. **Train-Test Split**
```python
X_train (105 samples), X_test (45 samples) → 70:30 split
y_train, y_test → Label/target untuk setiap sampel
random_state=42 → Untuk reproducibility
```

### 5. **Model KNN - Versi Dasar**
```python
KNeighborsClassifier() dengan default parameter (n_neighbors=5)
Training → Prediksi → Evaluasi
```

**Hasil Evaluasi**:
- Akurasi, Confusion Matrix, Classification Report

### 6. **Model KNN - Dengan Custom Parameter**
```python
n_neighbors=5
weights='distance'
metric='euclidean'
```

**Perbandingan** hasil antara model dasar dan custom parameter.

### 7. **GridSearchCV - Hyperparameter Tuning**
Mencari kombinasi parameter terbaik menggunakan cross-validation (CV=5):

**Parameter Grid**:
```python
n_neighbors: [3, 5, 7]
weights: ['uniform', 'distance']
metric: ['euclidean', 'manhattan']
Total kombinasi: 3 × 2 × 2 = 12 kombinasi
```

**Scoring Metrics**:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-Score (macro) ← Metrik utama untuk refit

**Output**:
- Tabel hasil cross-validation (CV results)
- Simpan hasil ke `hasil_gridsearch_knn.csv` atau `.xlsx`
- Parameter terbaik dari GridSearch

### 8. **Evaluasi Model Terbaik pada Test Set**
Menggunakan best_model dari GridSearchCV:
- Akurasi
- Presisi
- Recall
- F1-Score
- Classification Report (per class)
- Confusion Matrix

### 9. **Prediksi Data Baru (Contoh 1)**
```python
Input: SepalLengthCm=6.0, SepalWidthCm=3.0, PetalLengthCm=4.5, PetalWidthCm=1.5
Output: Prediksi jenis bunga iris
```

### 10. **Prediksi dengan K=5 (Contoh 2) - Penjelasan Detail**
```python
Input: SepalLengthCm=6.0, SepalWidthCm=5.0, PetalLengthCm=3.5, PetalWidthCm=1.0

Proses:
1. Hitung jarak dari data baru ke semua data training
2. Ambil 5 tetangga terdekat (K=5)
3. Lihat class dari 5 tetangga terdekat
4. Prediksi = class yang paling banyak (majority voting)

Output:
- Prediksi jenis bunga iris
- 5 tetangga terdekat dengan jarak mereka
- Hasil voting (berapa vote untuk setiap class)
```

### 11. **Visualisasi: Sebelum dan Sesudah Prediksi**
Dua subplot untuk menampilkan:

**Plot Kiri (SEBELUM)**:
- Semua 105 sampel data training
- Dibedakan dengan warna berdasarkan species
- Legenda: Iris-setosa (merah), Iris-versicolor (hijau), Iris-virginica (biru)

**Plot Kanan (SESUDAH)**:
- Data training dengan alpha rendah (background)
- Data baru ditampilkan dengan marker berlian oranye (◆)
- 5 tetangga terdekat ditampilkan dengan marker bintang besar (★)
- Garis putus menghubungkan data baru ke setiap tetangga
- Lingkaran merah putus menunjukkan radius pencarian (max distance)
- Judul menampilkan hasil prediksi

---

## 📈 Metrik Evaluasi

### Akurasi (Accuracy)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Persentase prediksi yang benar
```

### Presisi (Precision)
```
Precision = TP / (TP + FP)
Dari yang diprediksi positif, berapa yang benar positif
```

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
Dari yang sebenarnya positif, berapa yang berhasil ditangkap
```

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean antara precision dan recall
```

### Confusion Matrix
```
Menampilkan:
- True Positive (TP): Diprediksi Positif, Sebenarnya Positif
- True Negative (TN): Diprediksi Negatif, Sebenarnya Negatif
- False Positive (FP): Diprediksi Positif, Sebenarnya Negatif
- False Negative (FN): Diprediksi Negatif, Sebenarnya Positif
```

---

## 🎯 Cara Kerja KNN (K-Nearest Neighbors)

### Algoritma KNN:
1. **Hitung Jarak** dari data baru ke semua data training menggunakan metrik jarak (Euclidean/Manhattan)
2. **Urutkan** data training berdasarkan jarak (ascending)
3. **Ambil K tetangga terdekat** (misalnya K=5 berarti 5 tetangga terdekat)
4. **Majority Voting** - Lihat class dari K tetangga, ambil class yang paling sering muncul
5. **Prediksi** = class yang paling banyak dari K tetangga

### Contoh dengan K=5:
```
Data Baru: (6, 5, 3.5, 1.0)

Jarak ke 5 tetangga terdekat:
1. Jarak 0.5 → Iris-versicolor
2. Jarak 0.7 → Iris-versicolor
3. Jarak 0.9 → Iris-versicolor
4. Jarak 1.2 → Iris-setosa
5. Jarak 1.5 → Iris-versicolor

Voting:
- Iris-versicolor: 4 votes
- Iris-setosa: 1 vote

Prediksi = Iris-versicolor (menang dengan 4 votes)
```

---

## 🛠️ Parameter KNN

### `n_neighbors` (K)
- Jumlah tetangga terdekat yang dipertimbangkan
- **Default**: 5
- **Nilai yang dicoba**: [3, 5, 7]
- **Trade-off**: 
  - K kecil → Model sensitif (high variance)
  - K besar → Model smooth (high bias)

### `weights`
- Cara memberikan weight pada setiap tetangga
- **Options**:
  - `'uniform'`: Semua tetangga memiliki weight yang sama
  - `'distance'`: Tetangga yang lebih dekat memiliki weight lebih tinggi
- **Dicoba**: ['uniform', 'distance']

### `metric` (Jarak)
- Metrik yang digunakan untuk menghitung jarak
- **Options**:
  - `'euclidean'`: Jarak Euclidean = √[(x₂-x₁)² + (y₂-y₁)²]
  - `'manhattan'`: Jarak Manhattan = |x₂-x₁| + |y₂-y₁|
- **Dicoba**: ['euclidean', 'manhattan']

---

## 📁 File Output

### `hasil_gridsearch_knn.csv` atau `.xlsx`
Hasil evaluasi dari GridSearchCV berisi:
- Kombinasi parameter yang diuji
- Rata-rata akurasi, presisi, recall, F1-score dari cross-validation
- Diurutkan berdasarkan F1-score (terbaik di atas)

---

## 💡 Tips Penggunaan

### 1. **Mengubah Parameter Model**
```python
# Buat model dengan parameter custom
knn_custom = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')
knn_custom.fit(X_train, y_train)
```

### 2. **Prediksi Data Baru**
```python
# Siapkan data baru dalam DataFrame dengan kolom yang sama
new_data = pd.DataFrame([[6.0, 3.0, 4.5, 1.5]], 
                        columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
prediksi = best_model.predict(new_data)
```

### 3. **Melihat Tetangga Terdekat**
```python
distances, indices = knn_model.kneighbors(new_data, n_neighbors=5)
```

### 4. **Mengubah Metrik Scoring**
Tambahkan/ubah di `scoring` dictionary dalam GridSearchCV

---

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| `pandas` | Manipulasi dan analisis data |
| `numpy` | Komputasi numerik |
| `matplotlib` | Visualisasi dasar |
| `seaborn` | Visualisasi statistik |
| `sklearn.neighbors` | KNN classifier |
| `sklearn.model_selection` | Train-test split, GridSearchCV, cross-validation |
| `sklearn.preprocessing` | Label encoding |
| `sklearn.metrics` | Metrik evaluasi (accuracy, precision, recall, F1, confusion matrix) |

---

## 🎓 Kesimpulan

Program ini mendemonstrasikan:
✅ Penyelesaian masalah klasifikasi multiclass (3 kelas)  
✅ Exploratory Data Analysis untuk memahami dataset  
✅ Hyperparameter tuning dengan GridSearchCV  
✅ Evaluasi model dengan multiple metrics  
✅ Interpretasi hasil prediksi dan visualisasi  
✅ Penjelasan algoritma KNN secara detail  

---

## 👤 Author
Program dibuat untuk pembelajaran Supervised Learning - Classification

---

## 📝 Catatan Penting

- **Random State**: Gunakan `random_state=42` agar hasil reproducible
- **Normalisasi**: Dataset Iris sudah dalam skala yang reasonable, tapi untuk dataset lain pertimbangkan normalisasi
- **Imbalanced Data**: Jika data tidak balanced, pertimbangkan `class_weight` atau resampling
- **Cross-Validation**: GridSearchCV menggunakan 5-fold CV untuk hasil yang lebih robust

---

**Last Updated**: April 2026
