import math
import re
import time
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Zа-яА-ЯіІїЇєЄ]+\b', text.lower())


def compute_tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)

    return {word: count / total for word, count in counts.items()}


def compute_tfidf_single(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    return {word: tf_val * idf.get(word, 0.0) for word, tf_val in tf.items()}


def process_chunk(texts_chunk: list[str]) -> tuple[list[dict[str, float]], dict[str, int]]:
    tf_list: list[dict[str, float]] = []
    partial_df: dict[str, int] = {}
    for text in texts_chunk:
        tokens = tokenize(text)
        tf_list.append(compute_tf(tokens))
        for word in set(tokens):
            partial_df[word] = partial_df.get(word, 0) + 1

    return tf_list, partial_df


def sequential_tfidf(texts: list[str]) -> list[dict[str, float]]:
    N = len(texts)

    tf_list, global_df = process_chunk(texts)

    idf = {word: math.log((1 + N) / (1 + freq)) + 1
           for word, freq in global_df.items()}

    return [compute_tfidf_single(tf, idf) for tf in tf_list]


def run_sequential(texts: list[str]) -> tuple[list[dict[str, float]], float]:
    start = time.perf_counter()
    result = sequential_tfidf(texts)

    return result, time.perf_counter() - start
