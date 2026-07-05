"""
========================================================================
automate_JonathanBenedictSirait.py
------------------------------------------------------------------------
Otomatisasi tahap PREPROCESSING data credit scoring.

File ini adalah hasil KONVERSI dari notebook eksperimen
(Eksperimen_JonathanBenedictSirait.ipynb). Langkah-langkahnya PERSIS SAMA, hanya
strukturnya diubah menjadi FUNGSI agar bisa dijalankan otomatis
(misalnya lewat GitHub Actions).

Cara pakai (dari dalam folder 'preprocessing'):
    python automate_JonathanBenedictSirait.py

Output:
    Folder 'credit_scoring_preprocessing/' berisi train.csv, test.csv,
    dan credit_scoring_clean.csv yang SIAP DILATIH.
========================================================================
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Seed global agar hasil selalu sama (reproducible)
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1) LOAD DATA
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Membaca dataset mentah (CSV) menjadi DataFrame."""
    df = pd.read_csv(path)
    print(f"[LOAD] Data dimuat dari '{path}' dengan ukuran {df.shape}")
    return df


# ----------------------------------------------------------------------
# 2) BERSIHKAN DATA (duplikat, outlier, missing value)
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus duplikat & outlier, lalu mengisi nilai kosong."""
    data = df.copy()

    # (a) Hapus baris duplikat
    before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"[CLEAN] Hapus duplikat: {before} -> {len(data)} baris")

    # (b) Hapus outlier tidak wajar (umur > 100, lama kerja > 60 tahun)
    before = len(data)
    data = data[(data["person_age"] <= 100) &
                (data["person_emp_length"].fillna(0) <= 60)].reset_index(drop=True)
    print(f"[CLEAN] Hapus outlier: {before} -> {len(data)} baris")

    # (c) Isi nilai kosong pada kolom numerik dengan median
    for col in ["person_emp_length", "loan_int_rate"]:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
    print(f"[CLEAN] Missing value diisi median. Sisa kosong: {data.isnull().sum().sum()}")

    return data


# ----------------------------------------------------------------------
# 3) ENCODING FITUR KATEGORIKAL
# ----------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah kolom teks/kategori menjadi angka."""
    data = df.copy()

    # (a) Kolom biner Y/N -> 0/1
    data["cb_person_default_on_file"] = data["cb_person_default_on_file"].map({"N": 0, "Y": 1})

    # (b) Kolom multi-kategori -> One-Hot Encoding
    cat_cols = ["person_home_ownership", "loan_intent", "loan_grade"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # (c) Ubah kolom hasil one-hot (True/False) menjadi 0/1
    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)

    print(f"[ENCODE] Encoding selesai. Jumlah kolom sekarang: {data.shape[1]}")
    return data


# ----------------------------------------------------------------------
# 4) SPLIT + SCALING
# ----------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame, target: str = "loan_status"):
    """Memisahkan fitur & target, split train/test, lalu standardisasi."""
    X = df.drop(target, axis=1)
    y = df[target]

    # Split 80% latih : 20% uji, jaga proporsi kelas (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Kolom numerik yang di-scaling
    scale_cols = ["person_age", "person_income", "person_emp_length", "loan_amnt",
                  "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

    scaler = StandardScaler()
    # PENTING: fit hanya di train agar tidak terjadi data leakage
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    print(f"[SCALE] Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# 5) SIMPAN HASIL
# ----------------------------------------------------------------------
def save_output(X_train, X_test, y_train, y_test, out_dir: str):
    """Menyimpan hasil preprocessing ke folder output."""
    os.makedirs(out_dir, exist_ok=True)

    train_df = X_train.copy(); train_df["loan_status"] = y_train.values
    test_df = X_test.copy();   test_df["loan_status"] = y_test.values

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df.to_csv(os.path.join(out_dir, "credit_scoring_clean.csv"), index=False)

    print(f"[SAVE] Hasil tersimpan di '{out_dir}/' (train.csv, test.csv, credit_scoring_clean.csv)")


# ----------------------------------------------------------------------
# FUNGSI UTAMA — merangkai seluruh pipeline
# ----------------------------------------------------------------------
def preprocess_pipeline(input_path: str, output_dir: str):
    """Menjalankan seluruh tahap preprocessing dari awal sampai akhir."""
    print("=" * 60)
    print("MULAI PREPROCESSING OTOMATIS")
    print("=" * 60)

    df = load_data(input_path)          # 1. load
    df = clean_data(df)                 # 2. bersihkan
    df = encode_features(df)            # 3. encoding
    X_train, X_test, y_train, y_test = split_and_scale(df)  # 4. split + scale
    save_output(X_train, X_test, y_train, y_test, output_dir)  # 5. simpan

    print("=" * 60)
    print("PREPROCESSING SELESAI — data siap dilatih!")
    print("=" * 60)


if __name__ == "__main__":
    # Argumen baris perintah agar path bisa diatur fleksibel (berguna di GitHub Actions)
    parser = argparse.ArgumentParser(description="Otomatisasi preprocessing credit scoring")
    parser.add_argument("--input", default="../namadataset_raw/credit_scoring_raw.csv",
                        help="Path ke file CSV mentah")
    parser.add_argument("--output", default="credit_scoring_preprocessing",
                        help="Folder tempat menyimpan hasil preprocessing")
    args = parser.parse_args()

    preprocess_pipeline(args.input, args.output)
