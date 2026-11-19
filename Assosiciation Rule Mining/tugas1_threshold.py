"Tugas 1 - Eksplorasi ambang batas min_support dan min_confidence."


import matplotlib.pyplot as plt
import pandas as pd

from common import (
    DATA_DIR,
    encode,
    header,
    load_transactions,
    mine,
    rule_label,
    rules_table,
    save_csv,
    save_fig,
)

MIN_SUP_MODUL = 0.15
MIN_CONF_MODUL = 0.5
CONF_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def main():
    transactions = load_transactions(
        DATA_DIR / "transactions_arm_praktikum.csv"
    )
    df = encode(transactions)
    print(f"Jumlah transaksi: {len(df)}, jumlah item: {df.shape[1]}")

    # ---------------------------------------------------------------- 1.1
    header("1.1  min_support 0.10 vs 0.15 (min_confidence = 0.5)")
    f15, r15 = mine(df, MIN_SUP_MODUL, MIN_CONF_MODUL)
    f10, r10 = mine(df, 0.10, MIN_CONF_MODUL)
    banding = pd.DataFrame(
        {
            "min_support": [0.15, 0.10],
            "jumlah_itemset": [len(f15), len(f10)],
            "jumlah_aturan": [len(r15), len(r10)],
        }
    )
    print(banding.to_string(index=False))

    # itemset & aturan yang MUNCUL hanya ketika min_support turun ke 0.10
    set15 = {frozenset(s) for s in f15["itemsets"]}
    baru = f10[~f10["itemsets"].apply(lambda s: frozenset(s) in set15)]
    print("\nItemset baru pada min_support = 0.10:")
    print(baru.assign(itemsets=baru["itemsets"].apply(lambda s: ", ".join(sorted(s))))
          .round(3).to_string(index=False))

    key15 = set(r15.apply(rule_label, axis=1)) if len(r15) else set()
    r10_baru = r10[~r10.apply(rule_label, axis=1).isin(key15)] if len(r10) else r10
    print("\nAturan baru pada min_support = 0.10:")
    print(rules_table(r10_baru).to_string(index=False))
    save_csv(banding, "tugas1_banding_min_support.csv")

    # ---------------------------------------------------------------- 1.2
    header("1.2  Naikkan min_confidence 0.5 -> 0.7 (min_support = 0.15)")
    _, r70 = mine(df, MIN_SUP_MODUL, 0.7)
    lolos = set(r70.apply(rule_label, axis=1)) if len(r70) else set()
    tabel = rules_table(r15)
    tabel["status_conf_0.7"] = tabel["aturan"].apply(
        lambda a: "LOLOS" if a in lolos else "GUGUR"
    )
    print(f"Aturan awal (conf 0.5): {len(r15)}  |  lolos conf 0.7: {len(r70)}  "
          f"|  gugur: {len(r15) - len(r70)}")
    print(tabel.to_string(index=False))
    print("\nAturan yang gugur:")
    print(tabel[tabel["status_conf_0.7"] == "GUGUR"]["aturan"].to_string(index=False))
    save_csv(tabel, "tugas1_status_conf07.csv")

    # ---------------------------------------------------------------- 1.3
    header("1.3  Sweep min_confidence (min_support = 0.15)")
    freq = f15
    hasil = []
    for c in CONF_SWEEP:
        _, r = mine(df, MIN_SUP_MODUL, c)
        hasil.append(
            {
                "min_confidence": c,
                "jumlah_aturan": len(r),
                "rata2_lift": round(r["lift"].mean(), 3) if len(r) else float("nan"),
            }
        )
        print(f"min_confidence={c:.1f}: {len(r):2d} aturan")
    sweep = pd.DataFrame(hasil)
    save_csv(sweep, "tugas1_sweep_min_confidence.csv")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sweep["min_confidence"], sweep["jumlah_aturan"], "s-", color="#C0392B")
    for x, y in zip(sweep["min_confidence"], sweep["jumlah_aturan"]):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7), ha="center")
    ax.axvline(MIN_CONF_MODUL, ls=":", color="gray")
    ax.set_xlabel("min_confidence")
    ax.set_ylabel("Jumlah aturan asosiasi")
    ax.set_title(f"Pengaruh min_confidence (min_support = {MIN_SUP_MODUL})")
    ax.grid(alpha=0.3)
    save_fig(fig, "tugas1_sweep_min_confidence.png")

    # sebagai pelengkap: sweep min_support (seperti Gambar 5 modul)
    sup_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    n_item, n_rule = [], []
    for s in sup_list:
        f, r = mine(df, s, MIN_CONF_MODUL)
        n_item.append(len(f))
        n_rule.append(len(r))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sup_list, n_item, "o-", label="Frequent itemset", color="#1E2761")
    ax.plot(sup_list, n_rule, "s-", label="Aturan asosiasi", color="#C0392B")
    ax.set_xlabel("min_support")
    ax.set_ylabel("Jumlah")
    ax.set_title("Pengaruh min_support (min_confidence = 0.5)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, "tugas1_sweep_min_support.png")

    # ---------------------------------------------------------------- 1.4
    header("1.4  Bahan diskusi trade-off lebih banyak aturan vs lebih sedikit aturan")
    print(
        f"- conf 0.5 -> {int(sweep.iloc[0]['jumlah_aturan'])} aturan; "
        f"conf 1.0 -> {int(sweep.iloc[-1]['jumlah_aturan'])} aturan.\n"
        "- Ambang rendah: aturan banyak, tetapi memuat pola lemah/kebetulan (noise)\n"
        "  dan lebih sulit dipilih untuk keputusan bisnis.\n"
        "- Ambang tinggi: aturan sedikit tapi lebih andal; risikonya pola menarik\n"
        "  yang jarang terjadi (support kecil) ikut terbuang.\n"
        "- Cek juga Lift: confidence tinggi belum tentu Lift tinggi.\n"
        "Tulis analisis akhir dengan kalimat kelompok sendiri di laporan."
    )


if __name__ == "__main__":
    main()

    # ----------------------------------------------------------------
    # 1.5 ANALISIS HASIL EKSPERIMEN
    header("1.5  Analisis Hasil Eksperimen")

    print("""
    ANALISIS 1: PENGARUH MIN_SUPPORT

    Berdasarkan hasil eksperimen, nilai min_support memengaruhi jumlah
    frequent itemset dan aturan asosiasi yang dihasilkan. Ketika
    min_support diturunkan dari 0.15 menjadi 0.10, sistem dapat menemukan
    lebih banyak pola kombinasi item karena itemset dengan frekuensi
    kemunculan yang lebih rendah ikut diperhitungkan.

    Sebaliknya, ketika min_support dinaikkan, hanya kombinasi item yang
    sering muncul dalam transaksi yang dapat dipertahankan. Hal ini
    membuat jumlah frequent itemset dan aturan asosiasi menjadi lebih
    sedikit.


    ANALISIS 2: PENGARUH MIN_CONFIDENCE

    Perubahan nilai min_confidence juga memengaruhi jumlah aturan
    asosiasi yang dihasilkan. Semakin tinggi nilai min_confidence,
    semakin ketat persyaratan suatu aturan untuk diterima. Aturan yang
    memiliki confidence di bawah ambang batas akan dinyatakan gugur.

    Pada eksperimen ini, peningkatan min_confidence dari 0.5 menjadi
    0.7 bertujuan untuk menyaring aturan yang memiliki tingkat
    kepercayaan lebih rendah. Dengan demikian, aturan yang tersisa
    memiliki hubungan asosiasi yang lebih kuat berdasarkan nilai
    confidence.


    ANALISIS 3: PENGARUH MIN_CONFIDENCE TERHADAP LIFT

    Nilai lift digunakan untuk mengetahui kekuatan hubungan antara
    dua item dalam aturan asosiasi. Lift lebih dari 1 menunjukkan
    adanya hubungan positif, lift sama dengan 1 menunjukkan bahwa
    kedua item cenderung independen, sedangkan lift kurang dari 1
    menunjukkan hubungan negatif.

    Oleh karena itu, aturan dengan confidence tinggi belum tentu
    memiliki lift yang tinggi. Kedua nilai tersebut perlu diperhatikan
    agar aturan asosiasi yang dipilih tidak hanya memiliki tingkat
    kepercayaan tinggi, tetapi juga menunjukkan hubungan yang menarik.


    ANALISIS 4: TRADE-OFF AMBANG BATAS

    Penggunaan ambang batas rendah menghasilkan lebih banyak aturan
    asosiasi sehingga peluang menemukan pola tersembunyi menjadi
    lebih besar. Namun, aturan yang dihasilkan berpotensi mengandung
    noise atau pola kebetulan yang kurang relevan.

    Sebaliknya, penggunaan ambang batas tinggi menghasilkan aturan
    yang lebih sedikit dan lebih selektif. Meskipun demikian, pola
    menarik yang jarang terjadi berisiko tidak ditemukan karena
    tidak memenuhi persyaratan minimum.

    Dengan demikian, pemilihan min_support dan min_confidence harus
    disesuaikan dengan karakteristik data dan tujuan analisis.
    Kombinasi ambang batas yang tepat diperlukan agar aturan yang
    dihasilkan tetap relevan, cukup banyak, dan memiliki kualitas
    yang baik.
    """)

    print("Analisis eksperimen selesai.")
