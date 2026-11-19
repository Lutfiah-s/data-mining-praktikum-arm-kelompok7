"Tugas 2 - Analisis mendalam aturan (leverage & conviction)"

import math

import pandas as pd

from common import DATA_DIR, encode, header, load_transactions, mine, rule_label, save_csv

MIN_SUP = 0.15
MIN_CONF = 0.5

# (antecedent, consequent) -> aturan yang dianalisis
PILIHAN = [
    (("Mie Instan",), ("Telur",)),
    (("Roti",), ("Susu",)),
]

# rekomendasi
REKOMENDASI = {
    (("Mie Instan",), ("Telur",)): (
        "Pelanggan yang membeli Mie Instan juga cenderung membeli Telur. "
        "Toko dapat menempatkan telur di dekat rak mie instan atau menawarkan "
        "bundling 'Paket Mie Instan + Telur' untuk meningkatkan jumlah "
        "produk yang dibeli dalam satu transaksi."
    ),

    (("Roti",), ("Susu",)): (
        "Pelanggan yang membeli Roti juga memiliki kecenderungan membeli Susu. "
        "Toko dapat menempatkan susu berdekatan dengan rak roti atau menawarkan "
        "bundling 'Paket Roti + Susu'. Pada aplikasi belanja online, toko juga "
        "dapat menampilkan rekomendasi produk 'Pelanggan yang membeli roti "
        "juga membeli susu'."
    ),
}

def support(df, items):
    return df[list(items)].all(axis=1).mean()


def hitung_manual(df, ant, cons):
    """Hitung semua ukuran langsung dari data (untuk verifikasi mlxtend)."""
    s_a, s_c = support(df, ant), support(df, cons)
    s_ac = support(df, ant + cons)
    conf = s_ac / s_a
    lift = conf / s_c
    leverage = s_ac - s_a * s_c
    conviction = math.inf if conf >= 1 else (1 - s_c) / (1 - conf)
    return dict(
        support_A=s_a, support_C=s_c, support_AC=s_ac,
        confidence=conf, lift=lift, leverage=leverage, conviction=conviction,
    )


def interpretasi(m):
    teks = []
    teks.append(
        f"Lift {m['lift']:.3f} -> "
        + ("korelasi positif (lebih sering dibeli bersama daripada kebetulan)."
           if m["lift"] > 1 else
           "independen/negatif (tidak ada keterkaitan berarti).")
    )
    teks.append(
        f"Leverage {m['leverage']:.3f} -> selisih proporsi transaksi bersama "
        f"dibanding jika independen ({m['leverage'] * 100:.1f}% dari seluruh transaksi)."
    )
    conv = m["conviction"]
    teks.append(
        "Conviction tak hingga -> aturan tidak pernah salah (confidence = 1)."
        if math.isinf(conv) else
        f"Conviction {conv:.3f} -> aturan {conv:.2f}x lebih jarang salah "
        "dibanding jika antecedent & consequent independen."
    )
    return teks

def interpretasi_bisnis(ant, cons, m):
    aturan = f"{', '.join(ant)} → {', '.join(cons)}"
    lift = m["lift"]
    confidence = m["confidence"] * 100
    rekomendasi = REKOMENDASI.get(
        (ant, cons), "Belum tersedia rekomendasi bisnis untuk aturan ini."
    )

    return (
        f"• {aturan} (Lift {lift:.3f}, "
        f"Confidence {confidence:.0f}%) — {rekomendasi}"
    )


def main():
    transactions = load_transactions(
        DATA_DIR / "transactions_arm_praktikum.csv"
    )
    df = encode(transactions)

    print(f"Jumlah transaksi: {len(df)}")
    print(f"Jumlah jenis item: {df.shape[1]}")
    freq, rules = mine(df, MIN_SUP, MIN_CONF)
    print(f"{len(freq)} itemset, {len(rules)} aturan (support {MIN_SUP}, conf {MIN_CONF})")

    baris = []
    for ant, cons in PILIHAN:
        header(f"Aturan: {', '.join(ant)} -> {', '.join(cons)}")
        m = hitung_manual(df, ant, cons)

        # cocokkan dengan keluaran mlxtend (leverage & conviction sudah dihitung di sana)
        cocok = rules[
            rules["antecedents"].apply(lambda s: set(s) == set(ant))
            & rules["consequents"].apply(lambda s: set(s) == set(cons))
        ]
        if len(cocok):
            r = cocok.iloc[0]
            for k in ("confidence", "lift", "leverage"):
                assert abs(r[k] - m[k]) < 1e-9, f"{k} manual != mlxtend"
            if not math.isinf(m["conviction"]):
                assert abs(r["conviction"] - m["conviction"]) < 1e-9
            print("Verifikasi: hitungan manual = keluaran mlxtend  [OK]")
        else:
            print("Peringatan: aturan ini tidak lolos ambang di mlxtend "
                  "(hanya hitungan manual yang ditampilkan).")

        print(f"support(A)={m['support_A']:.3f}  support(C)={m['support_C']:.3f}  "
              f"support(A,C)={m['support_AC']:.3f}")
        print(f"confidence={m['confidence']:.3f}  lift={m['lift']:.3f}  "
              f"leverage={m['leverage']:.3f}  conviction={m['conviction']:.3f}")
        print("\nInterpretasi ukuran:")
        for t in interpretasi(m):
            print("  -", t)
        print("\nInterpretasi Bisnis:")
        print(interpretasi_bisnis(ant, cons, m))
        baris.append(
            {"aturan": f"{', '.join(ant)} -> {', '.join(cons)}",
             **{k: round(v, 4) for k, v in m.items()}}
        )

        header("Interpretasi Bisnis dari Aturan Terpilih")

    print("\nAturan asosiasi yang dianalisis dapat diterjemahkan "
          "menjadi rekomendasi bisnis sebagai berikut:\n")

    for ant, cons in PILIHAN:
        m = hitung_manual(df, ant, cons)
        print(interpretasi_bisnis(ant, cons, m))
        print()

    print(
        "Perlu diingat bahwa aturan asosiasi menunjukkan korelasi, "
        "bukan hubungan sebab-akibat. Oleh karena itu, rekomendasi "
        "bisnis yang dihasilkan sebaiknya divalidasi lebih lanjut, "
        "misalnya melalui pengujian A/B sebelum diterapkan secara luas."
    )
    
if __name__ == "__main__":
    main()
