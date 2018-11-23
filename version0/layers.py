import torch
from torch import nn
import torch.nn.functional as F

from allennlp.nn import util


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class MemoryAnsPointer(nn.Module):
    def __init__(self, x_size, y_size, hidden_size, hop=1, dropout_rate=0, normalize=True):
        super(MemoryAnsPointer, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.hop = hop
        self.dropout_rate = dropout_rate
        self.FFNs_start = nn.ModuleList()
        self.SFUs_start = nn.ModuleList()
        self.FFNs_end = nn.ModuleList()
        self.SFUs_end = nn.ModuleList()
        for i in range(self.hop):
            self.FFNs_start.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_start.append(SFU(y_size, 2 * hidden_size))
            self.FFNs_end.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_end.append(SFU(y_size, 2 * hidden_size))
        self.yesno_predictor = FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 3, dropout_rate)

    def forward(self, x, y, x_mask, y_mask):
        z_s = y[:, -1, :].unsqueeze(1)  # [B, 1, I]
        z_e = None
        s = None
        e = None
        p_s = None
        p_e = None

        for i in range(self.hop):
            z_s_ = z_s.repeat(1, x.size(1), 1)  # [B, S, I]
            s = self.FFNs_start[i](torch.cat([x, z_s_, x * z_s_], 2)).squeeze(2)
            # s.data.masked_fill_(x_mask.data, -float('inf'))
            # p_s = F.softmax(s, dim=1)  # [B, S]
            p_s = util.masked_softmax(s, x_mask, dim=1)
            u_s = p_s.unsqueeze(1).bmm(x)  # [B, 1, I]
            z_e = self.SFUs_start[i](z_s, u_s)  # [B, 1, I]
            z_e_ = z_e.repeat(1, x.size(1), 1)  # [B, S, I]
            e = self.FFNs_end[i](torch.cat([x, z_e_, x * z_e_], 2)).squeeze(2)
            # e.data.masked_fill_(x_mask.data, -float('inf'))
            # p_e = F.softmax(e, dim=1)  # [B, S]
            p_e = util.masked_softmax(e, x_mask, dim=1)
            u_e = p_e.unsqueeze(1).bmm(x)  # [B, 1, I]
            z_s = self.SFUs_end[i](z_e, u_e)
        yesno = self.yesno_predictor(torch.cat([x, z_e_, x * z_e_], 2))
        if self.normalize:
            # if self.training:
            # In training we output log-softmax for NLL
            # p_s = F.log_softmax(s, dim=1)  # [B, S]
            p_s = util.masked_log_softmax(s, x_mask, dim=1)
            # p_e = F.log_softmax(e, dim=1)  # [B, S]
            p_e = util.masked_log_softmax(e, x_mask, dim=1)
            p_yesno = F.log_softmax(yesno, dim=2)
        # else:
        # ...Otherwise 0-1 probabilities
        # p_s = F.softmax(s, dim=1)  # [B, S]
        # p_e = F.softmax(e, dim=1)  # [B, S]
        # p_yesno = F.softmax(yesno, dim=2)
        else:
            p_s = s.exp()
            p_e = e.exp()
            p_yesno = yesno.exp()
        return p_s, p_e, p_yesno


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
        * o_i = sum(alpha_j * y_j) for i in X
        * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size):
        super(SeqAttnMatch, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        # self.linear1 = nn.Linear(input_size, input_size)

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        x_pro = self.linear1(x)
        x_pro = F.relu(x_pro)
        y_pro = self.linear1(y)
        y_pro = F.relu(y_pro)

        # b * len1 * len2
        scores = x_pro.bmm(y_pro.transpose(2, 1))

        # b * len1 * len2
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        alpha = util.masked_softmax(scores, y_mask, dim=2)

        # b * len1 * hdim
        matched_seq = alpha.bmm(y)
        return matched_seq


class SelfAttnMatch(nn.Module):
    def __init__(self, input_size):
        super(SelfAttnMatch, self).__init__()

    def forward(self, x, x_mask):
        """
            Args:
                x: batch * len1 * dim1
                x_mask: batch * len1 (1 for padding, 0 for true)
            Output:
                matched_seq: batch * len1 * dim1
        """
        scores = x.bmm(x.transpose(2, 1))
        x_len = x.size(1)
        for i in range(x_len):
            scores[:, i, i] = 0

        x_mask = x_mask.unsqueeze(1).expand(scores.size())

        alpha = util.masked_softmax(scores, x_mask)

        matched_seq = alpha.bmm(x)
        return matched_seq


class SFU(nn.Module):
    """
    Semantic Fusion Unit
        The ouput vector is expected to not only retrieve correlative information from fusion vectors,
        but also retain partly unchange as the input vector
    """

    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = F.tanh(self.linear_r(r_f))
        g = F.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o
