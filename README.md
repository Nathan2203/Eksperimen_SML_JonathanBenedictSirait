# Eksperimen SML — Credit Scoring (Kriteria 1)

Repository ini berisi tahap **eksperimen & otomatisasi preprocessing** untuk proyek akhir
"Membangun Sistem Machine Learning" (Dicoding).

## Studi Kasus
Memprediksi apakah seorang peminjam akan **gagal bayar** (`loan_status`: 1 = gagal bayar, 0 = lancar)
berdasarkan profil peminjam & informasi pinjaman.

## Struktur Folder
```
Eksperimen_SML_JonathanBenedictSirait/
├── .github/workflows/preprocessing.yml   # Workflow CI preprocessing (Advance)
├── namadataset_raw/
│   └── credit_scoring_raw.csv            # Dataset MENTAH
└── preprocessing/
    ├── Eksperimen_JonathanBenedictSirait.ipynb        # Notebook eksperimen (EDA + preprocessing)
    ├── automate_JonathanBenedictSirait.py             # Otomatisasi preprocessing (Skilled)
    └── credit_scoring_preprocessing/     # Dataset hasil preprocessing (siap dilatih)
```

## Cara Menjalankan Preprocessing Manual
```bash
cd preprocessing
python automate_JonathanBenedictSirait.py
```

## Tingkatan yang Dipenuhi
- **Basic**: eksplorasi manual di notebook (data loading, EDA, preprocessing).
- **Skilled**: `automate_JonathanBenedictSirait.py` untuk preprocessing otomatis.
- **Advance**: GitHub Actions menjalankan preprocessing otomatis tiap ada push.
