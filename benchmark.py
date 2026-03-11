import os
import random
import statistics

from tfidf_sequential import run_sequential
from tfidf_parallel import run_parallel
from vocabulary import VOCAB

SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
WORKER_COUNTS = [2, 4, 8, 10, 12, 16]
WORDS_PER_DOC = 500
RUNS = 20


def generate_corpus(n_docs: int, words_per_doc: int = 500, seed: int = 42) -> list[str]:
    rng = random.Random(seed)

    return [" ".join(rng.choices(VOCAB, k=words_per_doc)) for _ in range(n_docs)]


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
        ("one document", ["only one document here with words"]),
        ("identical documents", ["hello world hello"] * 5),
        ("1000 docs x 500 words", generate_corpus(1_000, 500)),
        ("5000 docs x 500 words", generate_corpus(5_000, 500)),
        ("10000 docs x 500 words", generate_corpus(10_000, 500)),
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


if __name__ == "__main__":
    print(f"CPU cores: {os.cpu_count()}\n")
    run_correctness_tests()
    run_benchmark()
