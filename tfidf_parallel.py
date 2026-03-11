import math
import os
import time
from multiprocessing import Pool

from tfidf_sequential import process_chunk, compute_tfidf_single


def _worker_process_chunk(texts_chunk: list[str]) -> tuple[list[dict[str, float]], dict[str, int]]:
    return process_chunk(texts_chunk)


def _worker_apply_idf(args: tuple[list[dict[str, float]], dict[str, float]]) -> list[dict[str, float]]:
    tf_chunk, idf = args

    return [compute_tfidf_single(tf, idf) for tf in tf_chunk]


def _split_into_chunks(lst: list, n: int) -> list[list]:
    size = max(1, math.ceil(len(lst) / n))

    return [lst[i: i + size] for i in range(0, len(lst), size)]


def parallel_tfidf(texts: list[str], n_workers: int | None = None) -> list[dict[str, float]]:
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    N = len(texts)
    chunks = _split_into_chunks(texts, n_workers)

    with Pool(processes=n_workers) as pool:

        phase1 = pool.map(_worker_process_chunk, chunks)

        global_df: dict[str, int] = {}
        all_tf_chunks: list[list[dict[str, float]]] = []
        for tf_list, partial_df in phase1:
            all_tf_chunks.append(tf_list)
            for word, cnt in partial_df.items():
                global_df[word] = global_df.get(word, 0) + cnt

        idf = {word: math.log((1 + N) / (1 + freq)) + 1
               for word, freq in global_df.items()}

        result_chunks = pool.map(
            _worker_apply_idf,
            [(chunk, idf) for chunk in all_tf_chunks],
        )

    return [doc for chunk in result_chunks for doc in chunk]


def run_parallel(texts: list[str], n_workers: int | None = None) -> tuple[list[dict[str, float]], float]:
    start = time.perf_counter()
    result = parallel_tfidf(texts, n_workers)

    return result, time.perf_counter() - start
