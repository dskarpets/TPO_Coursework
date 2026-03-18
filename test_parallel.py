from tfidf_sequential import run_sequential
from tfidf_parallel import run_parallel
from vocabulary import VOCAB
import random

WORKER_COUNTS = [2, 4, 8, 10, 12, 16]


def generate_corpus(n_docs, words_per_doc=500, seed=42):
    rng = random.Random(seed)
    return [" ".join(rng.choices(VOCAB, k=words_per_doc)) for _ in range(n_docs)]


def generate_unequal_corpus(n_docs, seed=42):
    rng = random.Random(seed)
    lengths = [rng.randint(50, 2000) for _ in range(n_docs)]
    return [" ".join(rng.choices(VOCAB, k=l)) for l in lengths]


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
    sizes = [1_000, 5_000, 10_000]

    print("Parallel vs sequential")
    for n in sizes:
        corpus = generate_corpus(n)
        seq_res, _ = run_sequential(corpus)
        print(f"  {n} docs:")
        for w in WORKER_COUNTS:
            par_res, _ = run_parallel(corpus, w)
            ok = results_are_equal(seq_res, par_res)
            print(f"    {w} workers  {'OK' if ok else 'FAIL'}")
