from allennlp.data.tokenizers import Token
from collections import Counter
from typing import List

import re
import string


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


def free_text_to_span(free_text, full_text):
    if free_text == "unknown":
        return "__NA__", -1, -1
    if normalize_answer(free_text) == "yes":
        return "__YES__", -1, -1
    if normalize_answer(free_text) == "no":
        return "__NO__", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls) - 1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i + j >= len(full_ls): break
            full_cnt[full_ls[i + j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert (best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i + best_j][1] + 1

    return full_text[char_i:char_j], char_i, char_j


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))


def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', s)])


def proc_train(ith, article):
    rows = []
    context = article['story']

    for j, (question, answers) in enumerate(zip(article['questions'], article['answers'])):
        gold_answer = answers['input_text']
        span_answer = answers['span_text']

        answer, char_i, char_j = free_text_to_span(gold_answer, span_answer)
        answer_choice = 0 if answer == '__NA__' else \
            1 if answer == '__YES__' else \
                2 if answer == '__NO__' else \
                    3  # Not a yes/no question

        if answer_choice == 3:
            answer_start = answers['span_start'] + char_i
            answer_end = answers['span_start'] + char_j
        else:
            answer_start, answer_end = -1, -1

        rationale = answers['span_text']
        rationale_start = answers['span_start']
        rationale_end = answers['span_end']

        q_text = question['input_text']
        if j > 0:
            q_text = article['answers'][j - 1]['input_text'] + " // " + q_text

        rows.append(
            (ith, q_text, answer, answer_start, answer_end, rationale, rationale_start, rationale_end, answer_choice))
    return rows, context
