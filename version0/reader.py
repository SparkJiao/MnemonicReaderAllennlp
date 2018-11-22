import json
import logging
import sys
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, \
    MetadataField, LabelField, ListField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


#
# commands.main()


@DatasetReader.register("coqa-bidaf-pp-yesno")
class SquadReader(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 scalar=0.8,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._scalar = scalar

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # debug
        f = open('debug.txt', 'w')

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")

        num = 0
        for story_id, paragraph_json in enumerate(dataset):
            paragraph = paragraph_json["story"]
            n_paragraph, padding = self.delete_leading_tokens_of_paragraph(paragraph)
            tokenized_paragraph = self._tokenizer.tokenize(n_paragraph)

            ind = 0
            for question_answer in paragraph_json['questions']:
                question_text = question_answer["input_text"].strip().replace("\n", "")
                # question_text = question_answer["input_text"].replace("\n", "")
                answer_texts = []

                tmp = paragraph_json["answers"][ind]['span_text']
                before = self.get_front_blanks(tmp, padding)
                input_text = paragraph_json["answers"][ind]['input_text'].strip().replace("\n", "")
                span_text = paragraph_json["answers"][ind]['span_text'].strip().replace("\n", "")
                start = paragraph_json["answers"][ind]['span_start'] + before
                end = start + len(span_text)

                if input_text.lower() == 'yes':
                    yesno = 'y'
                    answer = span_text
                elif input_text.lower() == 'no':
                    yesno = 'n'
                    answer = span_text
                elif input_text.lower() == 'unknown':
                    answer = n_paragraph[0]
                    yesno = 'x'
                    start = 0
                    end = 0
                else:
                    yesno = 'x'
                    begin = span_text.find(input_text)
                    if begin != -1:
                        start = start + begin
                        end = start + len(input_text)
                        num += 1
                    answer = input_text

                answer_texts.append(answer)

                span_starts = list()
                span_starts.append(start)

                span_ends = list()
                span_ends.append(end)

                if "additional_answers" in paragraph_json:
                    additional_answers = paragraph_json["additional_answers"]
                    for key in additional_answers:
                        tmp = additional_answers[key][ind]["span_text"]
                        # answer = tmp.strip().replace("\n", "")
                        before = self.get_front_blanks(tmp, padding)
                        start = additional_answers[key][ind]["span_start"] + before
                        span_text = additional_answers[key][ind]["span_text"].strip().replace("\n", "")
                        input_text = additional_answers[key][ind]["input_text"].strip().replace("\n", "")
                        end = start + len(span_text)

                        if input_text.lower() == 'yes':
                            answer = span_text
                        elif input_text.lower() == 'no':
                            answer = span_text
                        elif input_text.lower() == 'unknown':
                            answer = n_paragraph[0]
                            start = 0
                            end = 0
                        else:
                            begin = span_text.find(input_text)
                            if begin != -1:
                                start = start + begin
                                end = start + len(input_text)
                                num += 1
                            answer = input_text

                        answer_texts.append(answer)
                        span_starts.append(start)
                        span_ends.append(end)

                ind += 1
                # print(paragraph)

                instance = self.text_to_instance(question_text,
                                                 n_paragraph,
                                                 zip(span_starts, span_ends),
                                                 answer_texts,
                                                 yesno,
                                                 tokenized_paragraph)
                # yesno dealing stop here 18.10.29
                yield instance

        logger.info('start and end changed num: %d' % num)

    def get_front_blanks(self, answer, padding):
        answer = answer.replace("\n", "")
        before = 0
        for i in range(len(answer)):
            if answer[i] == ' ':
                before += 1
            else:
                break
        return before - padding

    def delete_leading_tokens_of_paragraph(self, paragraph):
        before = 0
        for i in range(len(paragraph)):
            if paragraph[i] == ' ' or paragraph[i] == '\n':
                before += 1
            else:
                break

        nparagraph = paragraph[before:]
        return nparagraph, before

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         yesno: str = None,
                         passage_tokens: List[Token] = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                   passage_tokens,
                                                   self._token_indexers,
                                                   passage_text,
                                                   token_spans,
                                                   answer_texts,
                                                   yesno)


def make_reading_comprehension_instance(question_tokens: List[Token],
                                        passage_tokens: List[Token],
                                        token_indexers: Dict[str, TokenIndexer],
                                        passage_text: str,
                                        token_spans: List[Tuple[int, int]] = None,
                                        answer_texts: List[str] = None,
                                        yesno: str = None,
                                        additional_metadata: Dict[str, Any] = None) -> Instance:
    additional_metadata = additional_metadata or {}
    fields: Dict[str, Field] = {}
    passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                'question_tokens': [token.text for token in question_tokens],
                'passage_tokens': [token.text for token in passage_tokens], }
    if answer_texts:
        metadata['answer_texts'] = answer_texts

    if token_spans:
        # There may be multiple answer annotations, so we pick the one that occurs the most.  This
        # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
        # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
        # all of the annotations.  This is why we have a separate official SQuAD metric calculation
        # (the "em" and "f1" metrics use the official script).
        candidate_answers: Counter = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        fields['span_start'] = IndexField(span_start, passage_field)
        fields['span_end'] = IndexField(span_end, passage_field)
        fields['yesno'] = LabelField(yesno, label_namespace="yesno_labels")

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)
    return Instance(fields)
