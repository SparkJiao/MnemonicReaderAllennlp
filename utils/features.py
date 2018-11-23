from allennlp.data.tokenizers import Token
from collections import Counter
from typing import List


def get_tf(tokens: List[Token]):
    counter = Counter([a.text.lower() for a in tokens])
    length = len(tokens)
    tf = []
    for i, a in enumerate(tokens):
        tf.append(counter[a.text.lower()] * 1.0 / length)
    return tf


def get_exact_match(tokens_a: List[Token], tokens_b: List[Token]):
    a_cased = {a.text for a in tokens_a}
    a_uncased = {a.text.lower() for a in tokens_a}
    a_lemma = {a.lemma_ for a in tokens_a}
    b_em_cased = ["0"] * len(tokens_b)
    b_em_uncased = ["0"] * len(tokens_b)
    b_in_lemma = ["0"] * len(tokens_b)
    for i in range(len(tokens_b)):
        if tokens_b[i].text in a_cased:
            b_em_cased[i] = "1"
        if tokens_b[i].text.lower() in a_uncased:
            b_em_uncased[i] = "1"
        if tokens_b[i].lemma_ in a_lemma:
            b_in_lemma[i] = "1"

    b_cased = {b.text for b in tokens_b}
    b_uncased = {b.text.lower() for b in tokens_b}
    b_lemma = {b.lemma_ for b in tokens_b}
    a_em_cased = ["0"] * len(tokens_a)
    a_em_uncased = ["0"] * len(tokens_a)
    a_in_lemma = ["0"] * len(tokens_a)
    for i in range(len(tokens_a)):
        if tokens_a[i].text in b_cased:
            a_em_cased[i] = "1"
        if tokens_a[i].text.lower() in b_uncased:
            a_em_uncased[i] = "1"
        if tokens_a[i].lemma_ in b_lemma:
            a_in_lemma[i] = "1"

    return a_em_cased, a_em_uncased, a_in_lemma, b_em_cased, b_em_uncased, b_in_lemma
