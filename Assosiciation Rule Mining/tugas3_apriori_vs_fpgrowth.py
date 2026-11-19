"Tugas 3 - Apriori vs FP-Growth pada min_support lebih rendah (0.05)"

import statistics
import time

import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth

from common import DATA_DIR, encode, header, load_transactions, save_csv, save_fig

REPEAT = 30
SUP_UTAMA = [0.15, 0.05]          # perbandingan yang diminta tugas
SUP_TAMBAHAN = [0.10, 0.03]       # pelengkap untuk melihat tren


def ukur(func, df, min_support):
    waktu, hasil = [], None
    func(df, min_support=min_support, use_colnames=True)  # pemanasan
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        hasil = func(df, min_support=min_support, use_colnames=True)
        waktu.append(time.perf_counter() - t0)
    return hasil, statistics.median(waktu) * 1000, min(waktu) * 1000


def main():
    transactions = load_transactions(
        DATA_DIR / "transactions_arm_praktikum.csv"
    )
    df = encode(transactions)

    print(f"Jumlah transaksi: {df.shape[0]}")
    print(f"Jumlah jenis item: {df.shape[1]}")
    print(f"Data: {df.shape[0]} transaksi x {df.shape[1]} item; "
          f"waktu = median dari {REPEAT} kali jalan")

    baris = []
    for s in sorted(set(SUP_UTAMA + SUP_TAMBAHAN), reverse=True):
        fa, ta, ta_min = ukur(apriori, df, s)
        ff, tf, tf_min = ukur(fpgrowth, df, s)
        sama = {frozenset(x) for x in fa["itemsets"]} == {frozenset(x) for x in ff["itemsets"]}
        baris.append(
            {
                "min_support": s,
                "jumlah_itemset_apriori": len(fa),
                "jumlah_itemset_fpgrowth": len(ff),
                "itemset_identik": sama,
                "apriori_ms": round(ta, 2),
                "fpgrowth_ms": round(tf, 2),
                "selisih_ms": round(ta - tf, 2),
                "rasio_apriori/fpgrowth": round(ta / tf, 2) if tf else float("nan"),
            }
        )

    tabel = pd.DataFrame(baris)
    header("Hasil perbandingan")
    print(tabel.to_string(index=False))
    save_csv(tabel, "tugas3_apriori_vs_fpgrowth.csv")

    # -------------------------------------------------- grafik utama (0.15 vs 0.05)
    utama = tabel[tabel["min_support"].isin(SUP_UTAMA)].sort_values(
        "min_support", ascending=False
    )
    fig, ax = plt.subplots(figsize=(7, 4.4))
    x = range(len(utama))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], utama["apriori_ms"], w, label="Apriori", color="#1E2761")
    b2 = ax.bar([i + w / 2 for i in x], utama["fpgrowth_ms"], w, label="FP-Growth", color="#F2C94C",
                edgecolor="gray")
    ax.bar_label(b1, fmt="%.2f", padding=2)
    ax.bar_label(b2, fmt="%.2f", padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"min_support={s:.2f}\n({n} itemset)"
         for s, n in zip(utama["min_support"], utama["jumlah_itemset_apriori"])]
    )
    ax.set_ylabel("Waktu eksekusi (ms, median)")
    ax.set_title("Apriori vs FP-Growth")
    ax.legend()
    save_fig(fig, "tugas3_waktu_eksekusi.png")

    # -------------------------------------------------- tren semua min_support
    t = tabel.sort_values("min_support")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(t["jumlah_itemset_apriori"], t["apriori_ms"], "o-", label="Apriori", color="#1E2761")
    ax.plot(t["jumlah_itemset_fpgrowth"], t["fpgrowth_ms"], "s-", label="FP-Growth", color="#C0392B")
    ax.set_xlabel("Jumlah frequent itemset yang ditemukan")
    ax.set_ylabel("Waktu eksekusi (ms, median)")
    ax.set_title("Waktu vs jumlah itemset")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "tugas3_tren_waktu.png")


    # -------------------------------------------------- ANALISIS HASIL
    header("Analisis Hasil Eksperimen")

    a = tabel[tabel["min_support"] == 0.15].iloc[0]
    b = tabel[tabel["min_support"] == 0.05].iloc[0]

    # Perubahan jumlah itemset
    perubahan = (
        (b["jumlah_itemset_apriori"] - a["jumlah_itemset_apriori"])
        / a["jumlah_itemset_apriori"] * 100
        if a["jumlah_itemset_apriori"] else 0
    )

    print("\n1. PENGARUH PENURUNAN MIN_SUPPORT")
    print(
        f"Pada min_support 0.15 ditemukan "
        f"{a['jumlah_itemset_apriori']} frequent itemset, "
        f"sedangkan pada min_support 0.05 ditemukan "
        f"{b['jumlah_itemset_apriori']} frequent itemset."
    )

    if b["jumlah_itemset_apriori"] > a["jumlah_itemset_apriori"]:
        print(
            f"Jumlah frequent itemset meningkat sebesar "
            f"{perubahan:.2f}%. Hal ini menunjukkan bahwa semakin "
            "rendah min_support, semakin banyak kombinasi item "
            "yang memenuhi persyaratan minimum support."
        )
    else:
        print(
            "Jumlah frequent itemset tidak mengalami peningkatan. "
            "Hal ini menunjukkan karakteristik pola pada dataset "
            "perlu diperhatikan lebih lanjut."
        )

    print("\n2. PERBANDINGAN WAKTU EKSEKUSI")

    for _, r in tabel[tabel["min_support"].isin(SUP_UTAMA)].iterrows():
        print(f"\nMin_support = {r['min_support']:.2f}")
        print(f"- Apriori   : {r['apriori_ms']:.2f} ms")
        print(f"- FP-Growth : {r['fpgrowth_ms']:.2f} ms")

        if r["rasio_apriori/fpgrowth"] > 1:
            print(
                f"FP-Growth lebih cepat sekitar "
                f"{r['rasio_apriori/fpgrowth']:.2f} kali "
                "dibandingkan Apriori."
            )
        elif r["rasio_apriori/fpgrowth"] < 1:
            print(
                f"Apriori lebih cepat sekitar "
                f"{1 / r['rasio_apriori/fpgrowth']:.2f} kali "
                "dibandingkan FP-Growth."
            )
        else:
            print("Kedua algoritma memiliki waktu eksekusi yang relatif sama.")

    print("\n3. ANALISIS PERUBAHAN WAKTU")

    for nama, kolom in [
        ("Apriori", "apriori_ms"),
        ("FP-Growth", "fpgrowth_ms")
    ]:
        awal = a[kolom]
        akhir = b[kolom]

        if akhir > awal:
            print(
                f"- {nama}: waktu eksekusi meningkat dari "
                f"{awal:.2f} ms menjadi {akhir:.2f} ms."
            )
        elif akhir < awal:
            print(
                f"- {nama}: waktu eksekusi menurun dari "
                f"{awal:.2f} ms menjadi {akhir:.2f} ms."
            )
        else:
            print(f"- {nama}: waktu eksekusi relatif tetap.")

    print(
        "Penurunan min_support berpotensi meningkatkan beban komputasi "
        "karena semakin banyak frequent itemset yang harus ditemukan. "
        "Namun, perubahan waktu aktual juga dipengaruhi oleh ukuran "
        "dataset dan overhead proses eksekusi."
    )

    print("\n4. PERBANDINGAN HASIL ITEMSET")

    if tabel["itemset_identik"].all():
        print(
            "Kedua algoritma menghasilkan frequent itemset yang identik "
            "pada seluruh variasi min_support yang diuji. Perbedaannya "
            "terletak pada strategi pencarian dan waktu eksekusi, "
            "bukan pada hasil frequent itemset."
        )
    else:
        print(
            "Ditemukan perbedaan hasil frequent itemset pada salah satu "
            "variasi min_support. Hasil tersebut perlu diperiksa lebih "
            "lanjut untuk memastikan kesesuaian kedua algoritma."
        )

    print("\n5. KESIMPULAN PERBANDINGAN ALGORITMA")

    rasio_15 = a["rasio_apriori/fpgrowth"]
    rasio_05 = b["rasio_apriori/fpgrowth"]

    if rasio_05 > rasio_15:
        print(
            "Keunggulan kecepatan FP-Growth dibandingkan Apriori "
            "semakin terlihat ketika min_support diturunkan menjadi "
            "0.05. Hal ini sejalan dengan karakteristik FP-Growth "
            "yang tidak perlu membangkitkan kandidat itemset."
        )
    elif rasio_05 < rasio_15:
        print(
            "Keunggulan relatif FP-Growth dibandingkan Apriori "
            "menurun pada min_support 0.05. Hal ini menunjukkan "
            "bahwa perbedaan performa kedua algoritma dipengaruhi "
            "oleh karakteristik dataset dan kondisi pengukuran."
        )
    else:
        print(
            "Perbandingan kecepatan relatif kedua algoritma tidak "
            "berubah antara min_support 0.15 dan 0.05."
        )

    print(
        "\nSecara umum, Apriori menggunakan pembangkitan kandidat "
        "dan pemindaian data berulang, sedangkan FP-Growth "
        "memanfaatkan struktur FP-Tree untuk menemukan frequent "
        "itemset tanpa pembangkitan kandidat. FP-Growth berpotensi "
        "lebih efisien terutama ketika jumlah kombinasi item "
        "semakin banyak."
    )

    print(
        "\nCatatan: Waktu eksekusi merupakan median dari "
        f"{REPEAT} kali pengukuran setelah pemanasan. "
        "Hasil pengujian dapat dipengaruhi oleh ukuran dataset, "
        "perangkat, dan overhead eksekusi."
    )

if __name__ == "__main__":
    main()
