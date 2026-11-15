"Tugas 4 - Studi kasus mandiri: seluruh pipeline pada dataset kelompok"

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from common import (
    DATA_DIR,
    ROOT,
    encode,
    header,
    label,
    load_transactions,
    mine,
    rule_label,
    rules_table,
    save_csv,
    save_fig,
)

MIN_TRX = 40
MIN_ITEM = 8


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=str(DATA_DIR / "transactions_kedai_kopi.csv"))
    p.add_argument("--min-support", type=float, default=0.08)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--top", type=int, default=8, help="jumlah aturan pada bar chart Lift")
    return p.parse_args()


def main():
    args = parse()
    path = Path(args.data)
    transactions = load_transactions(path)
    df = encode(transactions)
    n, k = df.shape

    header("1. Ringkasan dataset")
    print(f"File: {path}\nTransaksi: {n}  |  Jenis item: {k}")
    if n < MIN_TRX or k < MIN_ITEM:
        raise SystemExit(
            f"Dataset belum memenuhi syarat modul (>= {MIN_TRX} transaksi dan >= {MIN_ITEM} item)."
        )
    cnt = Counter(i for t in transactions for i in t)
    print("\nFrekuensi item (bantu menentukan min_support):")
    for item, f in cnt.most_common():
        print(f"  {item:20s}: {f:3d} (support = {f / n:.3f})")

    header(f"2. Apriori (min_support={args.min_support}) & aturan (min_confidence={args.min_confidence})")
    freq, rules = mine(df, args.min_support, args.min_confidence)
    print(f"Frequent itemset: {len(freq)}  |  Aturan: {len(rules)}")
    if len(rules) == 0:
        raise SystemExit("Tidak ada aturan. Turunkan --min-support atau --min-confidence.")
    tabel = rules_table(rules, extra_cols=("leverage", "conviction"))
    print(tabel.to_string(index=False))
    save_csv(tabel, "tugas4_rules.csv")

    
    # -------------------------------------------------- 3. VISUALISASI
    header("3. Visualisasi")

    # Urutkan aturan berdasarkan Lift tertinggi
    rules_sorted = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # 3.1 Scatter plot Support vs Confidence
    sizes = (rules["lift"] - rules["lift"].min() + 0.3) * 160

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        rules["support"],
        rules["confidence"],
        s=sizes,
        c=rules["lift"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="#1E2761"
    )

    fig.colorbar(sc, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support vs Confidence Seluruh Aturan")
    ax.grid(alpha=0.3)
    save_fig(fig, "tugas4_scatter.png")

    # 3.2 Bar chart Top aturan berdasarkan Lift
    top = rules_sorted.head(args.top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(top) + 1.5))
    bars = ax.barh(
        top.apply(rule_label, axis=1),
        top["lift"],
        color="#1E2761"
    )

    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.axvline(
        1.0,
        ls="--",
        color="red",
        lw=1,
        label="Lift = 1 (independen)"
    )

    ax.set_xlabel("Lift")
    ax.set_title(f"{len(top)} Aturan Asosiasi dengan Lift Tertinggi")
    ax.legend(loc="lower right")
    save_fig(fig, "tugas4_lift_top.png")

    # -------------------------------------------------- 4. INTERPRETASI BISNIS
    header("4. Interpretasi Bisnis dari 3 Aturan Lift Tertinggi")

    print(
        "Tiga aturan dengan nilai Lift tertinggi dari dataset kelompok "
        "dapat diterjemahkan menjadi rekomendasi bisnis berikut:\n"
    )

    for i, r in rules_sorted.head(3).iterrows():
        a = label(r["antecedents"])
        c = label(r["consequents"])

        lift = r["lift"]
        confidence = r["confidence"]
        support_value = r["support"]

        print(
            f"{i + 1}. {a} → {c} "
            f"(Lift {lift:.3f}, Confidence {confidence:.0%})"
        )

        if lift > 1:
            print(
                f"   Produk {a} dan {c} memiliki hubungan asosiasi positif. "
                f"Pelanggan yang membeli {a} cenderung lebih sering membeli "
                f"{c} dibandingkan kondisi pembelian secara independen."
            )
        elif lift == 1:
            print(
                f"   Produk {a} dan {c} cenderung independen, sehingga "
                "tidak ditemukan peningkatan peluang pembelian bersama."
            )
        else:
            print(
                f"   Produk {a} dan {c} memiliki hubungan asosiasi negatif. "
                "Keduanya cenderung lebih jarang dibeli bersama."
            )

        print(
            f"   Support sebesar {support_value:.3f} menunjukkan bahwa "
            f"sekitar {support_value:.1%} dari seluruh transaksi "
            "memuat kombinasi produk tersebut."
        )

        print(
            f"   Rekomendasi: Kedai kopi dapat menempatkan produk {c} "
            f"di dekat produk {a}, menawarkan paket bundling, atau "
            f"menampilkan rekomendasi {c} ketika pelanggan memilih {a}."
        )

        print()

    print(
        "Perlu diingat bahwa aturan asosiasi menunjukkan korelasi, "
        "bukan hubungan sebab-akibat. Oleh karena itu, rekomendasi "
        "bisnis sebaiknya divalidasi lebih lanjut melalui pengujian "
        "A/B sebelum diterapkan secara luas."
    )

    # -------------------------------------------------- 5. ANALISIS LEVERAGE DAN CONVICTION
    header("5. Analisis Leverage dan Conviction")

    for i, r in rules_sorted.head(3).iterrows():
        a = label(r["antecedents"])
        c = label(r["consequents"])

        leverage = r["leverage"]
        conviction = r["conviction"]

        print(f"\nAturan {i + 1}: {a} → {c}")

        print(f"- Leverage: {leverage:.4f}")

        if leverage > 0:
            print(
                "  Kombinasi produk lebih sering muncul dibandingkan "
                "frekuensi yang diharapkan jika kedua produk independen."
            )
        elif leverage < 0:
            print(
                "  Kombinasi produk lebih jarang muncul dibandingkan "
                "frekuensi yang diharapkan jika kedua produk independen."
            )
        else:
            print("  Tidak terdapat selisih frekuensi dari kondisi independen.")

        print(f"- Conviction: {conviction:.4f}")

        if conviction == float("inf"):
            print(
                "  Confidence mencapai 100%, sehingga tidak ditemukan "
                "transaksi yang memuat antecedent tanpa consequent."
            )
        elif conviction > 1:
            print(
                "  Aturan memiliki kecenderungan positif dan lebih jarang "
                "mengalami kesalahan prediksi dibandingkan kondisi independen."
            )
        elif conviction == 1:
            print(
                "  Aturan tidak menunjukkan peningkatan kekuatan prediksi "
                "dibandingkan kondisi independen."
            )
        else:
            print(
                "  Aturan menunjukkan kecenderungan negatif dalam "
                "memprediksi consequent."
            )

    # -------------------------------------------------- 6. ANALISIS KESELURUHAN PIPELINE
    header("6. Analisis Keseluruhan Pipeline")

    print("\nA. ANALISIS DATASET")
    print(
        f"Dataset kelompok memiliki {n} transaksi dan {k} jenis item. "
        "Data tersebut digunakan sebagai dasar proses pencarian pola "
        "asosiasi untuk mengetahui kombinasi produk yang sering dibeli "
        "secara bersamaan."
    )

    print("\nB. ANALISIS FREQUENT ITEMSET")
    print(
        f"Dengan menggunakan min_support sebesar {args.min_support}, "
        f"diperoleh {len(freq)} frequent itemset. "
        "Frequent itemset merupakan kombinasi item yang memenuhi "
        "ambang batas minimum support."
    )

    print("\nC. ANALISIS ATURAN ASOSIASI")
    print(
        f"Dengan menggunakan min_confidence sebesar "
        f"{args.min_confidence}, diperoleh {len(rules)} aturan asosiasi. "
        "Aturan tersebut menunjukkan hubungan keterkaitan antara "
        "produk yang terdapat dalam transaksi."
    )

    print("\nD. ANALISIS KUALITAS ATURAN")
    rata_lift = rules["lift"].mean()
    rata_conf = rules["confidence"].mean()
    aturan_positif = (rules["lift"] > 1).sum()

    print(f"- Rata-rata Lift       : {rata_lift:.3f}")
    print(f"- Rata-rata Confidence : {rata_conf:.3f}")
    print(f"- Aturan dengan Lift > 1: {aturan_positif} dari {len(rules)} aturan")

    print("\nE. KESIMPULAN")
    print(
        f"Berdasarkan hasil proses Association Rule Mining, "
        f"diperoleh {len(freq)} frequent itemset dan {len(rules)} "
        "aturan asosiasi dari dataset kelompok."
    )

    if aturan_positif > 0:
        print(
            f"Sebanyak {aturan_positif} aturan memiliki Lift lebih dari 1, "
            "yang menunjukkan adanya asosiasi positif antarproduk. "
            "Aturan dengan Lift tertinggi dapat dijadikan dasar "
            "untuk menyusun strategi bundling, penataan produk, "
            "dan rekomendasi pembelian."
        )
    else:
        print(
            "Tidak ditemukan aturan dengan Lift lebih dari 1. "
            "Oleh karena itu, pola asosiasi positif perlu ditinjau "
            "kembali melalui penyesuaian ambang batas atau evaluasi dataset."
        )

    print(
        "\nPemilihan nilai min_support dan min_confidence perlu "
        "mempertimbangkan jumlah aturan yang dihasilkan serta "
        "kualitas hubungan antarproduk. Nilai ambang batas yang "
        "terlalu rendah berpotensi menghasilkan banyak aturan "
        "yang kurang relevan, sedangkan nilai yang terlalu tinggi "
        "dapat menghilangkan pola menarik dalam transaksi."
    )

    print("\nSeluruh proses Association Rule Mining selesai.")

    print(
        f"\nParameter yang digunakan:\n"
        f"- Min Support    : {args.min_support}\n"
        f"- Min Confidence: {args.min_confidence}\n"
        f"- Jumlah Top Rule: {args.top}"
    )

    print("\nFile hasil yang dihasilkan:")
    print("- tugas4_rules.csv")
    print("- tugas4_scatter.png")
    print("- tugas4_lift_top.png")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
