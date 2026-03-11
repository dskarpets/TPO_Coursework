import math
import re
from collections import Counter

from tfidf_sequential import run_sequential

docs = [
    "parallel calculation in python",
    "multiprocessing in python",
    "parallel and multiprocessing in general programming",
]


def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


all_tokens = [tokenize(d) for d in docs]
print("Tokenization")
for i, t in enumerate(all_tokens):
    print(f"  doc{i}: {t}")

print("\nCalculate TF")
for i, t in enumerate(all_tokens):
    all_tf = []
for i, tokens in enumerate(all_tokens):
    total = len(tokens)
    counts = Counter(tokens)
    tf = {w: c / total for w, c in counts.items()}
    all_tf.append(tf)
    print(f"  doc{i} (|d|={total}):")
    for w, v in sorted(tf.items()):
        c = counts[w]
        print(f"    {w:10s}  {c}/{total} = {v:.6f}")

N = len(docs)
df = {}
for tokens in all_tokens:
    for w in set(tokens):
        df[w] = df.get(w, 0) + 1

print(f"\nCalculate IDF,  N={N}")
idf = {}
for w, f in sorted(df.items()):
    val = math.log((1 + N) / (1 + f)) + 1
    idf[w] = val
    print(f"  {w:10s}  df={f}  log({1 + N}/{1 + f})+1 = {val:.6f}")

print("\nCalculate TF-IDF")
manual_results = []
for i, tf in enumerate(all_tf):
    tfidf = {w: tf[w] * idf[w] for w in tf}
    manual_results.append(tfidf)
    print(f"  doc{i}:")
    for w, v in sorted(tfidf.items(), key=lambda x: -x[1]):
        print(f"    {w:10s}  {tf[w]:.6f} * {idf[w]:.6f} = {v:.6f}")

print("\nComparison with tfidf_sequential")
algo_results, _ = run_sequential(docs)

all_ok = True
for i, (manual, algo) in enumerate(zip(manual_results, algo_results)):
    print(f"  doc{i}:")
    for w in sorted(manual):
        m = manual[w]
        a = algo.get(w, -1)
        match = abs(m - a) < 1e-9
        if not match:
            all_ok = False
        print(f"    {w:10s}  manual={m:.6f}  algo={a:.6f}  {'OK' if match else 'MISMATCH'}")

print()
print("All results match." if all_ok else "Error! Results do not match")
