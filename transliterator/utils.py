import re
from math import exp
import heapq

import operator
from typing import List, Tuple, Generator, Set
from functools import reduce

INPUT_REGEXP = r"^[a-z' ]*$"
INPUT_REGEXP_COMPILED = re.compile(INPUT_REGEXP)


class InvalidInputError(Exception):
    pass


def unite_batch_hyps_by_words(
    all_results: List[str], scores: List[float], n_hyps: int
) -> List[List[Tuple[str, float]]]:
    batch_hyps: List[List[Tuple[str, float]]] = []
    word_hyps: List[Tuple[str, float]] = []
    for idx, (translit, score) in enumerate(zip(all_results, scores)):
        word_hyps.append((translit, score))
        if idx % n_hyps == n_hyps - 1:
            batch_hyps.append(word_hyps)
            word_hyps = []
    return batch_hyps


def split_sentences_into_words(src: List[str]) -> Tuple[List[str], List[int]]:
    prepared_words: List[str] = []
    nums_of_words: List[int] = []
    for line in src:
        if not INPUT_REGEXP_COMPILED.match(line):
            raise InvalidInputError(f"Line {line} cannot be translited. Line must satisfy this regexp: {INPUT_REGEXP}")
        sentence_words = line.split()
        prepared_words.extend(sentence_words)
        nums_of_words.append(len(sentence_words))
    return prepared_words, nums_of_words


def get_sentence_by_combination(hyps: List[List[Tuple[str, float]]], combination: Tuple[int, ...]) -> str:
    return " ".join([item[combination[pos]][0] for pos, item in enumerate(hyps)])


def get_new_combinations(
    hyps: List[List[Tuple[str, float]]],
    curr_best: Tuple[float, Tuple[int, ...]],
    n_words: int,
    n_hyps: int,
    seen: Set[Tuple[int, ...]],
) -> Generator[Tuple[float, Tuple[int, ...]], None, None]:
    curr_best_combination = curr_best[1]
    for i in range(n_words):
        if n_hyps > curr_best_combination[i] + 1:
            combination = tuple(value if j != i else value + 1 for j, value in enumerate(curr_best_combination))
        else:
            continue
        if combination not in seen:
            diff_best_curr = hyps[i][combination[i] - 1][1] - hyps[i][combination[i]][1]
            curr_sum = curr_best[0] + diff_best_curr
            curr_value = (curr_sum, combination)
            yield curr_value


def get_top_multiword_hyps(hyps: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    if len(hyps) == 0:
        return [("", 0.0)]
    n_words = len(hyps)
    n_hyps = len(hyps[0])
    top_sum = reduce(
        operator.add,
        [-item[0][1] for item in hyps],
    )
    top_combination = tuple(0 for _ in range(n_words))
    init_sol = (top_sum, top_combination)
    heap_list = [init_sol]
    seen = {top_combination}
    best_hyps = []
    for _ in range(n_hyps):
        curr_best = heapq.heappop(heap_list)
        best_hyps.append(curr_best)
        for curr_value in get_new_combinations(
            hyps=hyps, curr_best=curr_best, n_words=n_words, n_hyps=n_hyps, seen=seen
        ):
            heapq.heappush(heap_list, curr_value)
            seen.add(curr_value[1])
    return [
        (get_sentence_by_combination(hyps=hyps, combination=combination), -pseudoscore)
        for pseudoscore, combination in best_hyps
    ]


def get_sentence_hyps(
    words_en: List[str], hyps: List[List[Tuple[str, float]]], idx: int, num_of_words_in_sent: int
) -> Tuple[str, List[Tuple[str, float]]]:
    sentence = []
    words_in_sentence_hyps = []
    for i in range(num_of_words_in_sent):
        word_hyps = hyps[idx + i]
        sentence.append(words_en[idx + i])
        words_in_sentence_hyps.append(sorted(word_hyps, key=lambda item: item[1], reverse=True))
    sentence_hyps = get_top_multiword_hyps(hyps=words_in_sentence_hyps)
    return " ".join(sentence), sentence_hyps


def clip_hyps_by_threshold(sentence_en: str, hyps: List[Tuple[str, float]], threshold=0.0) -> List[Tuple[str, float]]:
    hyps_passed_threshold = []
    words_num = len(sentence_en.split())
    for hyp, score in hyps:
        if exp(score) > threshold**words_num:
            hyps_passed_threshold.append((hyp, score))
    return hyps_passed_threshold


def postprocess(
    words_en: List[str],
    hyps: List[List[Tuple[str, float]]],
    nums_of_words: List[int],
    threshold: float = 0.0,
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    idx = 0
    processed_translits = []
    for num in nums_of_words:
        sentence, sentence_hyps = get_sentence_hyps(words_en=words_en, hyps=hyps, idx=idx, num_of_words_in_sent=num)
        idx += num
        hyps_passed_threshold = clip_hyps_by_threshold(sentence_en=sentence, hyps=sentence_hyps, threshold=threshold)
        processed_translits.append((sentence, hyps_passed_threshold))
    return processed_translits
