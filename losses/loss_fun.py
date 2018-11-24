import torch
from torch.nn.functional import gumbel_softmax


def get_f1_matrix(span_start: torch.Tensor, span_end: torch.Tensor, passgae_length: int):
    """
    return the f1 value matrix from span_start and span_end with passage_length
    :param span_start: batch_size
    :param span_end: batch_size
    :param passgae_length: batch_size
    :return: f1_matrix: batch_size, passage_Length, passage_length
    """
    batch_size = span_start.size(0)
    f1_matrix = span_start.new_zeros((batch_size, passgae_length, passgae_length), dtype=torch.float)
    for i in range(passgae_length):
        for j in range(i, passgae_length, 1):
            f1_matrix[:, i, j] = cal_f1(torch.tensor([i], dtype=torch.float, device=span_start.device),
                                        torch.tensor([j], dtype=torch.float, device=span_start.device),
                                        span_start, span_end)
    return f1_matrix


def test_get_f1_matrix():
    span_start = torch.tensor([2, 3])
    span_end = torch.tensor([5, 9])
    f1_matrix = get_f1_matrix(span_start, span_end, 12)
    print(f1_matrix)


def evidence_f1_loss(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor):
    """
    sum(p_i * p_j * F1(i, j)), 0 <= i <= j < passage_length
    :param p_start: batch_size * passage_length(masked 0)
    :param p_end: batch_size * passage_length(masked 0)
    :param span_start: batch_size
    :param span_end: batch_size
    :return: loss
    """
    batch_size = p_start.size(0)
    passage_length = p_start.size(1)
    f1_matrix = get_f1_matrix(span_start, span_end, passage_length)
    span_prob = torch.bmm(p_start.unsqueeze(2), p_end.unsqueeze(2).transpose(2, 1))
    f1 = (span_prob * f1_matrix).reshape(batch_size, -1)
    loss = torch.sum(f1, 1)
    return loss


def cal_f1(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor):
    """
    :param p_start: 1
    :param p_end: 1
    :param span_start: batch_size
    :param span_end: batch_size
    :return: f1
    """
    batch_size = span_end.size(0)
    # batch_size
    span_start = torch.tensor(span_start, dtype=torch.float, device=span_start.device)
    span_end = torch.tensor(span_end, dtype=torch.float, device=span_end.device)
    start = torch.max(torch.cat([p_start.repeat(batch_size).unsqueeze(1), span_start.unsqueeze(1)], dim=1), dim=1)[0]
    end = torch.min(torch.cat([p_end.repeat(batch_size).unsqueeze(1), span_end.unsqueeze(1)], dim=1), dim=1)[0]
    num_same = end - start + 1
    precision = num_same / (p_end - p_start + 1)
    recall = num_same / (span_end - span_start + 1)
    f1 = (2 * precision * recall) / (precision + recall)
    for i in range(batch_size):
        if num_same[i] <= 0:
            f1[i] = 0
    return f1


def test_cal_f1():
    p_start = torch.tensor([3.])
    p_end = torch.tensor([5.])
    span_start = torch.tensor([2, 3])
    span_end = torch.tensor([5, 7])
    f1 = cal_f1(p_start, p_end, span_start, span_end)
    print(f1)


def answer_f1_loss(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor,
                   span_end: torch.Tensor):
    """

    :param p_start: batch_size * passage_length
    :param p_end: batch_size * passage_length
    :param span_start:  batch_size
    :param span_end:  batch_size
    :return:
    """
    passage_length = p_start.size(1)
    batch_size = p_start.size(0)
    # batch_size * passage_length * 1
    start_prob = p_start.unsqueeze(2)
    end_prob = p_end.unsqueeze(2)
    # batch_size * (passage_length * passage_length)
    span_prob = torch.bmm(start_prob, end_prob.transpose(2, 1)).reshape(batch_size, -1)
    # batch_size * (passage_length * passage_length) * 1
    span_prob_gumbel = gumbel_softmax(span_prob, hard=True).unsqueeze(-1)
    # batch_size * 1 * (p * p)
    f1_matrix = get_f1_matrix(span_start, span_end, passage_length).reshape(batch_size, -1).unsqueeze(1)
    return torch.bmm(f1_matrix, span_prob_gumbel)
