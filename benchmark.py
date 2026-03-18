import os
import random
import statistics

from tfidf_sequential import run_sequential
from tfidf_parallel import run_parallel
from vocabulary import VOCAB

SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
WORKER_COUNTS = [2, 4, 8, 10, 12, 16]
WORDS_PER_DOC = 1_000
RUNS = 20
WORD_COUNTS = [100, 500, 1_000, 2_000, 5_000]
DOCS_FOR_WORD_BENCH = 5_000


def generate_corpus(n_docs: int, words_per_doc: int = 2_000, seed: int = 42) -> list[str]:
    rng = random.Random(seed)

    return [" ".join(rng.choices(VOCAB, k=words_per_doc)) for _ in range(n_docs)]


def generate_unequal_corpus(n_docs: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    lengths = [rng.randint(50, 5_000) for _ in range(n_docs)]
    return [" ".join(rng.choices(VOCAB, k=l)) for l in lengths]


def measure(func, *args, runs: int = RUNS) -> float:
    func(*args)
    times = [func(*args)[1] for _ in range(runs)]
    return statistics.mean(times)


def results_are_equal(a: list[dict], b: list[dict], tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        if set(da) != set(db):
            return False
        for w in da:
            if abs(da[w] - db[w]) > tol:
                return False
    return True


def run_correctness_tests() -> None:
    print("CORRECTNESS TESTS")
    print("-" * 50)
    cases = [
        ("smoke test (3 docs)",
         ["the cat sat on the mat",
          "the dog sat on the log",
          "the cat and the dog are friends"]),
        ("one document",           ["only one document here with words"]),
        ("identical documents",    ["hello world hello"] * 5),
        ("1000 docs x 2000 words", generate_corpus(1_000, 2_000)),
        ("5000 docs x 2000 words", generate_corpus(5_000, 2_000)),
        ("10000 docs x 2000 words",generate_corpus(10_000, 2_000)),
        ("unequal doc lengths (1000 docs)", generate_unequal_corpus(1_000)),
    ]
    all_ok = True
    for name, texts in cases:
        seq_res, _ = run_sequential(texts)
        par_res, _ = run_parallel(texts)
        ok = results_are_equal(seq_res, par_res)
        if not ok:
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'}  {name}")
    print()
    print("All tests passed." if all_ok else "Error: results do not match!")
    print()


def run_benchmark() -> None:
    corpuses = {n: generate_corpus(n, WORDS_PER_DOC) for n in SIZES}

    print("SEQUENTIAL ALGORITHM")
    print("-" * 35)
    print(f"{'docs':>8}  {'time (s)':>10}")
    print("-" * 35)

    seq_times: dict[int, float] = {}
    for n in SIZES:
        t = measure(run_sequential, corpuses[n])
        seq_times[n] = t
        print(f"{n:>8}  {t:>10.3f}")
    print()

    max_cpu = os.cpu_count() or 8
    workers = [w for w in WORKER_COUNTS if w <= max_cpu * 2]

    for w in workers:
        print(f"PARALLEL ALGORITHM  {w} workers")
        print("-" * 45)
        print(f"{'docs':>8}  {'time (s)':>10}  {'speedup':>9}")
        print("-" * 45)
        for n in SIZES:
            t_par = measure(run_parallel, corpuses[n], w)
            sp = seq_times[n] / t_par if t_par > 0 else float("inf")
            print(f"{n:>8}  {t_par:>10.3f}  {sp:>9.2f}x")
        print()


def run_words_per_doc_benchmark() -> None:
    best_workers = min(os.cpu_count() or 4, 12)
    print(f"EFFECT OF DOCUMENT SIZE  ({DOCS_FOR_WORD_BENCH} docs, {best_workers} workers)")
    print("-" * 55)
    print(f"{'words/doc':>10}  {'seq (s)':>10}  {'par (s)':>10}  {'speedup':>9}")
    print("-" * 55)
    for wpd in WORD_COUNTS:
        corpus = generate_corpus(DOCS_FOR_WORD_BENCH, wpd)
        t_seq = measure(run_sequential, corpus)
        t_par = measure(run_parallel, corpus, best_workers)
        sp = t_seq / t_par if t_par > 0 else float("inf")
        print(f"{wpd:>10}  {t_seq:>10.3f}  {t_par:>10.3f}  {sp:>9.2f}x")
    print()


def run_unequal_benchmark() -> None:
    best_workers = min(os.cpu_count() or 4, 12)
    print(f"EQUAL VS UNEQUAL DOCUMENT LENGTHS  (5000 docs, {best_workers} workers)")
    print("-" * 55)
    print(f"{'corpus':>10}  {'seq (s)':>10}  {'par (s)':>10}  {'speedup':>9}")
    print("-" * 55)
    equal = generate_corpus(5_000, 2_000)
    unequal = generate_unequal_corpus(5_000)
    for label, corpus in [("equal", equal), ("unequal", unequal)]:
        t_seq = measure(run_sequential, corpus)
        t_par = measure(run_parallel, corpus, best_workers)
        sp = t_seq / t_par if t_par > 0 else float("inf")
        print(f"{label:>10}  {t_seq:>10.3f}  {t_par:>10.3f}  {sp:>9.2f}x")
    print()


if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}\n")
    run_correctness_tests()
    run_benchmark()
    run_words_per_doc_benchmark()
    run_unequal_benchmark()
