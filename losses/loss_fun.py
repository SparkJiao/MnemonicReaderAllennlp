import torch


def evidence_loss(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor,
                  passage_length: int):
    """
    :param p_start: batch_size * passage_length
    :param p_end: batch_size * passage_length
    :param span_start: batch_size
    :param span_end: batch_size
    :param passage_length: batch_size
    :return: loss
    """
    batch_size = p_start.size(0)
    loss = p_start.new_zeros(batch_size, dtype=torch.float)
    for i in range(batch_size):
        for j in range(passage_length):
            for k in range(j, passage_length, 1):
                loss[i] = p_start[i] * p_start[k] * cal_f1(
                    torch.tensor([j], dtype=torch.float, device=span_end[i].device),
                    torch.tensor([k], dtype=torch.float, device=span_end[i].device),
                    span_start[i], span_end[i])
    return loss


def cal_f1(p_start: torch.Tensor, p_end: torch.Tensor, span_start: torch.Tensor, span_end: torch.Tensor):
    """
    :param p_start:
    :param p_end:
    :param span_start:
    :param span_end:
    :return: f1
    """
    # batch_size
    start = max(p_start, span_start)
    end = min(p_end, span_end)
    num_same = end - start + 1
    precision = num_same / (p_end - p_start + 1)
    recall = num_same / (span_end - span_start + 1)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
