import logging
from typing import Any, Dict, List, Optional
from overrides import overrides
import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1

from torch import nn
from version0 import layers
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("model")
class BidirectionalAttentionFlow(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 char_rnn: Seq2SeqEncoder,
                 hops: int,
                 hidden_dim: int,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._encoding_dim = phrase_layer.get_output_dim()
        # self._stacked_brnn = PytorchSeq2SeqWrapper(
        #     StackedBidirectionalLstm(input_size=self._encoding_dim, hidden_size=hidden_dim,
        #                              num_layers=3, recurrent_dropout_probability=0.2))
        self._char_rnn = char_rnn

        self.hops = hops

        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(hops):
            # interactive aligner
            self.interactive_aligners.append(layers.SeqAttnMatch(self._encoding_dim))
            self.interactive_SFUs.append(layers.SFU(self._encoding_dim, 3 * self._encoding_dim))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(self._encoding_dim))
            self.self_SFUs.append(layers.SFU(self._encoding_dim, 3 * self._encoding_dim))
            # aggregating
            self.aggregate_rnns.append(PytorchSeq2SeqWrapper(nn.LSTM(
                input_size=self._encoding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                dropout=0.2,
                bidirectional=True,
                batch_first=True)))

        # Memmory-based Answer Pointer
        self.mem_ans_ptr = layers.MemoryAnsPointer(
            x_size=self._encoding_dim,
            y_size=self._encoding_dim,
            hidden_size=hidden_dim,
            hop=hops,
            dropout_rate=0.2,
            normalize=True
        )

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_yesno_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                yesno: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ

        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))

        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        token_emb_q, char_emb_q, question_word_features = torch.split(embedded_question, [300, 100, 40], dim=2)
        token_emb_c, char_emb_c, passage_word_features = torch.split(embedded_passage, [300, 100, 40], dim=2)

        char_features_q = self._char_rnn(char_emb_q, question_lstm_mask)
        char_features_c = self._char_rnn(char_emb_c, passage_lstm_mask)

        emb_question = torch.cat([token_emb_q, char_features_q, question_word_features], dim=2)
        emb_passage = torch.cat([token_emb_c, char_features_c, passage_word_features], dim=2)

        encoded_question = self._dropout(self._phrase_layer(emb_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(emb_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # c_check = self._stacked_brnn(encoded_passage, passage_lstm_mask)
        # q = self._stacked_brnn(encoded_question, question_lstm_mask)
        c_check = encoded_passage
        q = encoded_question
        for i in range(self.hops):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, question_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check,
                                                     torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, passage_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, passage_mask)

        # Predict
        start_scores, end_scores, yesno_scores = self.mem_ans_ptr.forward(c_check, q, passage_mask, question_mask)

        best_span, yesno_predict, loc = self.get_best_span(start_scores, end_scores, yesno_scores)

        output_dict = {
            "span_start_logits": start_scores,
            "span_end_logits": end_scores,
            "best_span": best_span
        }

        # Compute the loss for training.
        if span_start is not None:
            loss = nll_loss(start_scores, span_start.squeeze(-1))
            self._span_start_accuracy(start_scores, span_start.squeeze(-1))
            loss += nll_loss(end_scores, span_end.squeeze(-1))
            self._span_end_accuracy(end_scores, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1))

            gold_span_end_loc = []
            span_end = span_end.view(batch_size).squeeze().data.cpu().numpy()
            for i in range(batch_size):
                gold_span_end_loc.append(max(span_end[i] + i * passage_length, 0))
            gold_span_end_loc = span_start.new(gold_span_end_loc)
            _yesno = yesno_scores.view(-1, 3).index_select(0, gold_span_end_loc).view(-1, 3)
            loss += nll_loss(_yesno, yesno.view(-1), ignore_index=-1)

            pred_span_end_loc = []
            for i in range(batch_size):
                pred_span_end_loc.append(max(loc[i], 0))
            predicted_end = span_start.new(pred_span_end_loc)
            _yesno = yesno_scores.view(-1, 3).index_select(0, predicted_end).view(-1, 3)
            self._span_yesno_accuracy(_yesno, yesno.squeeze(-1))

            output_dict['loss'] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
            output_dict['yesno'] = yesno_predict
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            "yesno": self._span_yesno_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        yesno_tags = [self.vocab.get_token_from_index(x, namespace="yesno_labels") for x in output_dict.pop("yesno")]
        output_dict['yesno'] = yesno_tags
        return output_dict

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor,
                      yesno_scores: torch.Tensor):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)
        yesno_predict = span_start_logits.new_zeros(batch_size, dtype=torch.long)
        loc = yesno_scores.new_zeros(batch_size, dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()
        yesno_logits = yesno_scores.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
                    yesno_predict[b] = int(np.argmax(yesno_logits[b, j]))
                    loc[b] = j + passage_length * b
        return best_word_span, yesno_predict, loc
