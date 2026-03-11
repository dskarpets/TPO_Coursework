from tfidf_sequential import run_sequential
from tfidf_parallel import run_parallel

docs = [
    "parallel calculation in python",
    "multiprocessing in python",
    "parallel and multiprocessing in general programming",
]

WORKER_COUNTS = [1, 2, 3, 4, 6, 8]


def results_are_equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        if set(da) != set(db):
            return False
        for w in da:
            if abs(da[w] - db[w]) > tol:
                return False
    return True


if __name__ == '__main__':
    seq_res, _ = run_sequential(docs)

    print("Comparison: parallel vs sequential")
    print(f"  {'workers':>8}  {'result':>6}")
    print("  " + "-" * 20)
    for w in WORKER_COUNTS:
        par_res, _ = run_parallel(docs, w)
        ok = results_are_equal(seq_res, par_res)
        print(f"  {w:>8}  {'OK' if ok else 'FAIL':>6}")

    print("\nDetailed comparison (workers=4)")
    par_res4, _ = run_parallel(docs, 4)
    all_ok = True
    for i, (s, p) in enumerate(zip(seq_res, par_res4)):
        print(f"  doc{i}:")
        for word in sorted(s):
            sv, pv = s[word], p[word]
            match = abs(sv - pv) < 1e-9
            if not match:
                all_ok = False
            print(f"    {word:20s}  seq={sv:.6f}  par={pv:.6f}  {'OK' if match else 'MISMATCH'}")

    print()
    print("All results match." if all_ok else "Error! Results do not match.")
