"""
============================================================
VISUALISASI KONVERGENSI: Hybrid GA + 2-Opt
============================================================
Membaca file history JSON yang dihasilkan oleh GA2Opt_7C.py
(folder: output_konvergensi/) dan memplot grafik konvergensi
nilai Best Z per generasi untuk pengujian verifikasi algoritma.

Output disimpan ke folder output_konvergensi/grafik/ dalam
format PNG resolusi 200 DPI.
============================================================
"""

import os
import json
import glob
import matplotlib.pyplot as plt


INPUT_DIR  = "output_konvergensi"
OUTPUT_DIR = os.path.join(INPUT_DIR, "grafik")


def plot_single_convergence(history_data, save_path):
    instance     = history_data["instance"]
    algorithm    = history_data["algorithm"]
    history      = history_data["best_cost_history"]
    best_z       = history_data["best_z"]
    num_cust     = history_data.get("num_customers", "-")

    generations = list(range(1, len(history) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(generations, history,
            color="#1f77b4", linewidth=2.0, marker="o", markersize=3,
            label="Best Z per Generasi")

    ax.axhline(y=best_z, color="red", linestyle="--", linewidth=1.2,
               label=f"Best Z Final = Rp {best_z:,.0f}")

    ax.set_xlabel("Generasi", fontsize=12)
    ax.set_ylabel("Best Z (Rp)", fontsize=12)
    ax.set_title(
        f"Grafik Konvergensi {algorithm} | {instance} | {num_cust} Pelanggan",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_convergence(all_data, save_path):
    """Gabungan semua instance dalam satu figure (untuk perbandingan)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10.colors
    for idx, hd in enumerate(all_data):
        history = hd["best_cost_history"]
        generations = list(range(1, len(history) + 1))
        ax.plot(generations, history,
                color=colors[idx % len(colors)], linewidth=1.8,
                label=f"{hd['instance']} (Best Z = Rp {hd['best_z']:,.0f})")

    ax.set_xlabel("Generasi", fontsize=12)
    ax.set_ylabel("Best Z (Rp)", fontsize=12)
    ax.set_title(
        "Grafik Konvergensi Gabungan | Hybrid GA + 2-Opt | Semua Instance",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Folder '{INPUT_DIR}' tidak ditemukan.")
        print(f"        Jalankan terlebih dahulu: python GA2Opt_7C.py")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pattern = os.path.join(INPUT_DIR, "history_HybridGA2Opt_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[ERROR] Tidak ada file history di '{INPUT_DIR}'.")
        print(f"        Pola dicari: {pattern}")
        raise SystemExit(1)

    print("=" * 80)
    print(f"  VISUALISASI KONVERGENSI HYBRID GA + 2-OPT")
    print(f"  Input  : {INPUT_DIR}/")
    print(f"  Output : {OUTPUT_DIR}/")
    print(f"  Jumlah file history : {len(files)}")
    print("=" * 80)

    all_data = []
    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)
        all_data.append(data)

        out_name = f"konvergensi_{data['algorithm']}_{data['instance']}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        plot_single_convergence(data, out_path)
        print(f"   |- [OK] {data['instance']:<6s} -> {out_path}")

    # Plot gabungan
    combined_path = os.path.join(OUTPUT_DIR, "konvergensi_gabungan.png")
    plot_combined_convergence(all_data, combined_path)
    print(f"   `- [OK] Gabungan -> {combined_path}")

    print("=" * 80)
    print(f"[SUKSES] {len(files)} grafik konvergensi + 1 grafik gabungan tersimpan.")
