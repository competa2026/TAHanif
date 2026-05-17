"""
============================================================
PERBANDINGAN: MILP-CBC (Eksak) vs HYBRID GA + 2-OPT (Metaheuristik)
Skala validasi: 7 Pelanggan | 10 Dataset Solomon
============================================================
TA: Hanif Hisyam Ramadhan (2026) - ITS Surabaya
Pembimbing: Prof. Budi Santosa, Ph.D.

Tujuan:
  Membandingkan Z optimal eksak (CBC) dengan solusi metaheuristik
  (GA+2Opt, 10 repetisi). Menghitung:
    - Best Gap (%)   = (GA_Best  - CBC) / CBC * 100
    - Avg  Gap (%)   = (GA_Avg   - CBC) / CBC * 100
    - Speedup waktu  = Waktu_CBC / Waktu_GA
    - Counter berapa instance GA mencapai BKS (gap <= 0.01%)

Input:
  - CSV CBC:  Hasil_MILP_CBC_MT_CVRPTW_PG_10_Dataset.csv
  - CSV GA :  FinalRun_HybridGA_2Opt_7Cust_10Rep_Parallel.csv
  Jika kedua CSV tidak ditemukan, script otomatis pakai data
  hardcoded dari output yang dikirim sebelumnya.

Output:
  - Tabel terminal (pandas to_string)
  - CSV: Perbandingan_CBC_vs_HybridGA_7Cust.csv
============================================================
"""

import os
import pandas as pd

# ---------------------------------------------------------
# DATA HARDCODED FALLBACK
# Diisi dari output txt yang dilampirkan user.
# Untuk RC105 CBC, dipakai nilai Z yang LEBIH RENDAH (better) sebagai BKS.
# ---------------------------------------------------------
CBC_HARDCODED = [
    {"Instance": "R101",  "CBC_Z": 2472852.53, "CBC_Truk": 2, "CBC_Waktu":  7.561, "CBC_Status": "Optimal"},
    {"Instance": "R111",  "CBC_Z": 1958069.39, "CBC_Truk": 1, "CBC_Waktu": 62.440, "CBC_Status": "Optimal"},
    {"Instance": "C101",  "CBC_Z": 2715870.13, "CBC_Truk": 1, "CBC_Waktu":  0.223, "CBC_Status": "Optimal"},
    {"Instance": "C105",  "CBC_Z": 2715870.13, "CBC_Truk": 1, "CBC_Waktu":  0.222, "CBC_Status": "Optimal"},
    {"Instance": "RC101", "CBC_Z": 2218959.69, "CBC_Truk": 1, "CBC_Waktu": 25.198, "CBC_Status": "Optimal"},
    {"Instance": "RC105", "CBC_Z": 2209492.10, "CBC_Truk": 1, "CBC_Waktu": 88.544, "CBC_Status": "Optimal"},
    {"Instance": "R201",  "CBC_Z": 2987039.12, "CBC_Truk": 1, "CBC_Waktu":  3.219, "CBC_Status": "Optimal"},
    {"Instance": "C201",  "CBC_Z": 4163402.66, "CBC_Truk": 1, "CBC_Waktu":  0.273, "CBC_Status": "Optimal"},
    {"Instance": "RC201", "CBC_Z": 3018959.69, "CBC_Truk": 1, "CBC_Waktu": 29.046, "CBC_Status": "Optimal"},
    {"Instance": "R205",  "CBC_Z": 2834978.65, "CBC_Truk": 1, "CBC_Waktu":  6.712, "CBC_Status": "Optimal"},
]

GA_HARDCODED = [
    {"Instance": "R101",  "GA_Best_Z": 2672853, "GA_Avg_Z": 2672853, "GA_Truk": 3.0, "GA_Waktu": 2.35, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "R111",  "GA_Best_Z": 2158069, "GA_Avg_Z": 2158069, "GA_Truk": 2.0, "GA_Waktu": 0.53, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "C101",  "GA_Best_Z": 2915870, "GA_Avg_Z": 2915870, "GA_Truk": 2.0, "GA_Waktu": 0.63, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "C105",  "GA_Best_Z": 2915870, "GA_Avg_Z": 2915870, "GA_Truk": 2.0, "GA_Waktu": 0.82, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "RC101", "GA_Best_Z": 2418960, "GA_Avg_Z": 2418960, "GA_Truk": 2.0, "GA_Waktu": 0.60, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "RC105", "GA_Best_Z": 2395461, "GA_Avg_Z": 2395461, "GA_Truk": 2.0, "GA_Waktu": 0.53, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "R201",  "GA_Best_Z": 3335972, "GA_Avg_Z": 3335972, "GA_Truk": 1.0, "GA_Waktu": 0.54, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "C201",  "GA_Best_Z": 4163403, "GA_Avg_Z": 4163403, "GA_Truk": 1.0, "GA_Waktu": 0.62, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "RC201", "GA_Best_Z": 3263960, "GA_Avg_Z": 3263960, "GA_Truk": 1.0, "GA_Waktu": 0.67, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
    {"Instance": "R205",  "GA_Best_Z": 3247483, "GA_Avg_Z": 3247483, "GA_Truk": 1.0, "GA_Waktu": 0.61, "GA_Valid": "10 / 10", "GA_Status": "Aman"},
]


# ---------------------------------------------------------
# CSV LOADERS (dengan auto-rename kolom)
# ---------------------------------------------------------
CBC_CSV = "Hasil_MILP_CBC_MT_CVRPTW_PG_10_Dataset.csv"
GA_CSV  = "FinalRun_HybridGA_2Opt_7Cust_10Rep_Parallel.csv"


def load_cbc():
    """Coba baca CBC CSV; jika tidak ada, pakai hardcoded."""
    if os.path.exists(CBC_CSV):
        df = pd.read_csv(CBC_CSV).rename(columns={
            "Z_Terbaik (Rp)"   : "CBC_Z",
            "Kendaraan_Dipakai": "CBC_Truk",
            "Waktu (s)"        : "CBC_Waktu",
            "Status"           : "CBC_Status",
        })
        print(f"[INFO] CBC dimuat dari '{CBC_CSV}' ({len(df)} baris)")
        return df[["Instance", "CBC_Z", "CBC_Truk", "CBC_Waktu", "CBC_Status"]]
    print(f"[INFO] '{CBC_CSV}' tidak ditemukan -> pakai data hardcoded CBC")
    return pd.DataFrame(CBC_HARDCODED)


def load_ga():
    """Coba baca GA CSV; jika tidak ada, pakai hardcoded."""
    if os.path.exists(GA_CSV):
        df = pd.read_csv(GA_CSV).rename(columns={
            "Best_Z (Rp)"           : "GA_Best_Z",
            "Rata2_Biaya_Z (Rp)"    : "GA_Avg_Z",
            "Rata2_Truk"            : "GA_Truk",
            "Rata2_Waktu (s)"       : "GA_Waktu",
            "Total_Valid"           : "GA_Valid",
            "Alasan_Mayoritas_Gagal": "GA_Status",
        })
        print(f"[INFO] GA dimuat dari '{GA_CSV}' ({len(df)} baris)")
        return df[["Instance", "GA_Best_Z", "GA_Avg_Z", "GA_Truk", "GA_Waktu", "GA_Valid", "GA_Status"]]
    print(f"[INFO] '{GA_CSV}' tidak ditemukan -> pakai data hardcoded GA")
    return pd.DataFrame(GA_HARDCODED)


# ---------------------------------------------------------
# BUILD COMPARISON TABLE
# ---------------------------------------------------------
def build_comparison(df_cbc, df_ga):
    df = pd.merge(df_cbc, df_ga, on="Instance", how="inner")

    df["Gap_Best (%)"] = ((df["GA_Best_Z"] - df["CBC_Z"]) / df["CBC_Z"]) * 100
    df["Gap_Avg (%)"]  = ((df["GA_Avg_Z"]  - df["CBC_Z"]) / df["CBC_Z"]) * 100
    df["Speedup"]      = df["CBC_Waktu"] / df["GA_Waktu"]

    def conclude(row):
        gb = row["Gap_Best (%)"]
        if abs(gb) <= 0.01:
            return "GA = BKS (optimal)"
        elif gb < 0:
            return f"GA LEBIH BAIK ({gb:+.2f}%) !"
        elif gb <= 1.0:
            return f"Near-optimal (+{gb:.2f}%)"
        elif gb <= 5.0:
            return f"Acceptable (+{gb:.2f}%)"
        else:
            return f"Suboptimal (+{gb:.2f}%)"

    df["Keterangan"] = df.apply(conclude, axis=1)
    return df


def format_display(df):
    disp = df.copy()
    disp["CBC_Z"]        = disp["CBC_Z"].apply(lambda x: f"Rp {x:>13,.0f}")
    disp["GA_Best_Z"]    = disp["GA_Best_Z"].apply(lambda x: f"Rp {x:>13,.0f}")
    disp["GA_Avg_Z"]     = disp["GA_Avg_Z"].apply(lambda x: f"Rp {x:>13,.0f}")
    disp["CBC_Waktu"]    = disp["CBC_Waktu"].apply(lambda x: f"{x:>7.2f}s")
    disp["GA_Waktu"]     = disp["GA_Waktu"].apply(lambda x: f"{x:>6.2f}s")
    disp["CBC_Truk"]     = disp["CBC_Truk"].apply(lambda x: f"{int(x)}")
    disp["GA_Truk"]      = disp["GA_Truk"].apply(lambda x: f"{x:.1f}")
    disp["Gap_Best (%)"] = disp["Gap_Best (%)"].apply(lambda x: f"{x:+.2f}%")
    disp["Gap_Avg (%)"]  = disp["Gap_Avg (%)"].apply(lambda x: f"{x:+.2f}%")
    disp["Speedup"]      = disp["Speedup"].apply(lambda x: f"{x:>5.1f}x")
    return disp


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 115)
    print("  PERBANDINGAN: MILP-CBC (EKSAK) vs HYBRID GA + 2-OPT (METAHEURISTIK)")
    print("  Skala: 7 Pelanggan | 10 Dataset Solomon | GA = 10 repetisi")
    print("=" * 115)

    df_cbc = load_cbc()
    df_ga  = load_ga()
    df_cmp = build_comparison(df_cbc, df_ga)
    df_disp = format_display(df_cmp)

    # ---- Tabel utama (kolom inti) ----
    cols_main = ["Instance", "CBC_Z", "CBC_Truk", "CBC_Waktu",
                 "GA_Best_Z", "GA_Avg_Z", "GA_Truk", "GA_Waktu", "GA_Valid",
                 "Gap_Best (%)", "Gap_Avg (%)", "Speedup", "Keterangan"]

    print("\n" + "-" * 145)
    print("  TABEL PERBANDINGAN HEAD-TO-HEAD")
    print("-" * 145)
    print(df_disp[cols_main].to_string(index=False))
    print("-" * 145)

    # ---- Ringkasan statistik ----
    n_total      = len(df_cmp)
    n_optimal    = (df_cmp["Gap_Best (%)"].abs() <= 0.01).sum()
    n_near_opt   = ((df_cmp["Gap_Best (%)"] > 0.01) & (df_cmp["Gap_Best (%)"] <= 1.0)).sum()
    n_acceptable = ((df_cmp["Gap_Best (%)"] > 1.0) & (df_cmp["Gap_Best (%)"] <= 5.0)).sum()
    n_subopt     = (df_cmp["Gap_Best (%)"] > 5.0).sum()
    avg_gap_best = df_cmp["Gap_Best (%)"].mean()
    avg_gap_avg  = df_cmp["Gap_Avg (%)"].mean()
    max_gap_best = df_cmp["Gap_Best (%)"].max()
    min_gap_best = df_cmp["Gap_Best (%)"].min()
    avg_speedup  = df_cmp["Speedup"].mean()
    med_speedup  = df_cmp["Speedup"].median()

    print("\n" + "=" * 115)
    print("  RINGKASAN STATISTIK")
    print("=" * 115)
    print(f"  Total instance dibandingkan          : {n_total}")
    print(f"  GA mencapai BKS (gap <= 0.01%)       : {n_optimal} / {n_total}")
    print(f"  GA near-optimal (0.01% < gap <= 1%)  : {n_near_opt} / {n_total}")
    print(f"  GA acceptable    (1% < gap <= 5%)    : {n_acceptable} / {n_total}")
    print(f"  GA suboptimal    (gap > 5%)          : {n_subopt} / {n_total}")
    print(f"  Rata-rata Gap (Best vs BKS)          : {avg_gap_best:+.2f}%")
    print(f"  Rata-rata Gap (Avg  vs BKS)          : {avg_gap_avg:+.2f}%")
    print(f"  Range Gap Best                       : [{min_gap_best:+.2f}%, {max_gap_best:+.2f}%]")
    print(f"  Rata-rata Speedup (T_CBC / T_GA)     : {avg_speedup:.1f}x")
    print(f"  Median Speedup                       : {med_speedup:.1f}x")
    print("=" * 115)

    # ---- Interpretasi otomatis ----
    print("\n" + "=" * 115)
    print("  INTERPRETASI UNTUK BAB VALIDASI TA")
    print("=" * 115)
    if n_optimal == n_total:
        print("  -> SEMPURNA: GA+2Opt mencapai solusi optimal di SEMUA instance.")
        print("     Ini bukti kuat bahwa metaheuristik MAMPU mereplikasi hasil eksak MILP")
        print("     pada skala validasi, dengan waktu komputasi jauh lebih cepat.")
    elif n_optimal + n_near_opt == n_total:
        pct_opt = 100.0 * n_optimal / n_total
        print(f"  -> SANGAT BAIK: {n_optimal} ({pct_opt:.0f}%) instance mencapai BKS persis;")
        print(f"     {n_near_opt} instance lainnya near-optimal (<=1% gap).")
        print(f"     Avg gap {avg_gap_best:.2f}% menunjukkan GA+2Opt konvergen mendekati optimal.")
    elif avg_gap_best <= 10.0:
        print(f"  -> WAJAR: rata-rata gap {avg_gap_best:.2f}% terhadap BKS.")
        print(f"     Untuk skala kecil (n=7) ini menunjukkan ruang improvement pada tuning")
        print(f"     parameter GA atau iterasi tambahan. Pada skala besar (n>=50) di mana CBC")
        print(f"     tidak tractable, GA tetap pilihan utama.")
    else:
        print(f"  -> PERLU TUNING: rata-rata gap {avg_gap_best:.2f}% relatif tinggi.")
        print(f"     Pertimbangkan: pop_size lebih besar, max_gen lebih lama, atau seeding")
        print(f"     awal dengan heuristik konstruktif (savings algorithm dll).")
    print(f"\n  Speedup median {med_speedup:.1f}x menunjukkan GA jauh lebih cepat per run,")
    print(f"  walau CBC tetap superior dari sisi guarantee optimality.")
    print("=" * 115)

    # ---- Simpan CSV ----
    out_csv = "Perbandingan_CBC_vs_HybridGA_7Cust.csv"
    df_cmp.to_csv(out_csv, index=False)
    print(f"\n[SAVED] Hasil perbandingan disimpan ke: '{out_csv}'")
    print(f"        (Kolom numerik untuk re-analysis; tampilan terformat hanya di terminal)")
