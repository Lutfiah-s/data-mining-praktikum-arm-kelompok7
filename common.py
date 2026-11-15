from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # simpan grafik ke file; aman dijalankan tanpa layar
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth  # noqa: E402
from mlxtend.preprocessing import TransactionEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA_MODUL = DATA_DIR / "transaksi_modul.csv"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_transactions(path=DATA_MODUL):
    """Baca file transaksi: satu baris = satu transaksi, item dipisah ';'.

    Baris kosong dan baris berawalan '#' diabaikan.
    """
    transactions = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip() and not ln.startswith("#")]

    # Format 2: CSV dengan header "TID,Items" dan item dipisah koma di kolom Items
    if lines and lines[0].replace(" ", "").lower().startswith("tid,"):
        import csv

        for row in csv.reader(lines[1:]):
            if len(row) < 2:
                continue
            items = [x.strip() for x in ",".join(row[1:]).split(",") if x.strip()]
            if items:
                transactions.append(items)
        return transactions

    # Format 1: satu baris = satu transaksi, item dipisah ';'
    for line in lines:
        items = [x.strip() for x in line.split(";") if x.strip()]
        if items:
            transactions.append(items)
    return transactions


def encode(transactions):
    """List of list -> DataFrame one-hot (True/False) via TransactionEncoder."""
    te = TransactionEncoder()
    arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_)


# --------------------------------------------------------------------------
# Mining
# --------------------------------------------------------------------------
def build_rules(freq, n_transactions, min_confidence):
    """Bentuk aturan asosiasi dari frequent itemset, urut berdasarkan Lift.

    Memakai pemanggilan seperti di modul; jika versi mlxtend terpasang
    meminta argumen `num_itemsets`, dicoba ulang dengan argumen tersebut.
    """
    if len(freq) == 0:
        return pd.DataFrame()
    try:
        try:
            rules = association_rules(
                freq, metric="confidence", min_threshold=min_confidence
            )
        except TypeError:
            rules = association_rules(
                freq,
                num_itemsets=n_transactions,
                metric="confidence",
                min_threshold=min_confidence,
            )
    except ValueError:  # mis. hanya ada 1-itemset -> tidak ada aturan
        return pd.DataFrame()
    if len(rules) == 0:
        return rules
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def mine(df, min_support, min_confidence, algo="apriori"):
    """Jalankan algoritma + pembentukan aturan. Return (freq_itemset, rules)."""
    func = apriori if algo == "apriori" else fpgrowth
    freq = func(df, min_support=min_support, use_colnames=True)
    freq = freq.sort_values("support", ascending=False).reset_index(drop=True)
    rules = build_rules(freq, len(df), min_confidence)
    return freq, rules


# --------------------------------------------------------------------------
# Tampilan
# --------------------------------------------------------------------------
def label(itemset):
    """frozenset -> 'A, B' (urut abjad)."""
    return ", ".join(sorted(itemset))


def rule_label(row):
    return f"{label(row['antecedents'])} -> {label(row['consequents'])}"


def rules_table(rules, extra_cols=()):
    """DataFrame ringkas siap cetak."""
    if len(rules) == 0:
        return pd.DataFrame(columns=["aturan", "support", "confidence", "lift"])
    t = pd.DataFrame(
        {
            "aturan": rules.apply(rule_label, axis=1),
            "support": rules["support"],
            "confidence": rules["confidence"],
            "lift": rules["lift"],
        }
    )
    for c in extra_cols:
        t[c] = rules[c]
    return t.round(3)


def header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[tersimpan] {path.relative_to(ROOT)}")
    return path


def save_csv(df, name):
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    print(f"[tersimpan] {path.relative_to(ROOT)}")
    return path
