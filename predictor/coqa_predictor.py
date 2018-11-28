import json
from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model


@Predictor.register('predictor')
class DialogQAPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm')

    def predict(self, jsonline: str) -> JsonDict:
        """
        Make a dialog-style question answering prediction on the supplied input.
        The supplied input json must contain a list of
        question answer pairs, containing question, answer, yesno, followup, id
        as well as the context (passage).
        Parameters
        ----------
        jsonline: ``str``
            A json line that has the same format as the quac data file.
        Returns
        ----------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json(json.loads(jsonline))

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects json that looks like the original quac data file.
        """
        paragraph_json = json_dict
        paragraph = paragraph_json["story"]
        tokenized_paragraph = self._tokenizer.tokenize(paragraph)
        metadata = dict()
        metadata['id'] = paragraph_json['id']
        ind = 0
        for question_answer in paragraph_json['questions']:
            question_text = question_answer["input_text"].strip().replace("\n", "")
            # question_text = question_answer["input_text"].replace("\n", "")
            answer_texts = []

            if paragraph_json["answers"][ind]['span_start'] == -1:
                start = 0
                end = 0
                answer = "CANNOTUNSWER"
            else:
                tmp = paragraph_json["answers"][ind]['span_text']
                before = self.get_front_blanks(tmp, 0)
                input_text = paragraph_json["answers"][ind]['input_text'].strip().replace('\n', '')
                span_text = paragraph_json["answers"][ind]['span_text'].strip().replace('\n', '')
                start = paragraph_json["answers"][ind]['span_start'] + before
                end = start + len(span_text)
                r_input_text = input_text.replace('\n', '').lower()

                begin = span_text.find(input_text)
                if begin != -1:
                    start = start + begin
                    end = start + len(input_text)
                    answer = input_text
                    yesno = 'x'
                else:
                    if r_input_text == 'yes':
                        yesno = 'y'
                    elif r_input_text == 'no':
                        yesno = 'n'
                    else:
                        yesno = 'x'
                    answer = span_text

            answer_texts.append(answer)

            span_starts = list()
            span_starts.append(start)

            span_ends = list()
            span_ends.append(end)

            if "additional_answers" in paragraph_json:
                additional_answers = paragraph_json["additional_answers"]
                for key in additional_answers:
                    if additional_answers[key][ind]['span_start'] == -1:
                        start = 0
                        end = 0
                        answer = 'CANNOTANSWER'
                    else:
                        tmp = additional_answers[key][ind]["span_text"]
                        # answer = tmp.strip().replace("\n", "")
                        before = self.get_front_blanks(tmp, 0)
                        start = additional_answers[key][ind]["span_start"] + before
                        span_text = additional_answers[key][ind]["span_text"].strip().replace('\n', '')
                        input_text = additional_answers[key][ind]["input_text"].strip().replace('\n', '')
                        end = start + len(span_text)
                        r_input_text = input_text.lower()

                        begin = span_text.find(input_text)
                        if begin != -1:
                            start = start + begin
                            end = start + len(input_text)
                            answer = input_text
                        else:
                            answer = span_text

                    answer_texts.append(answer)
                    span_starts.append(start)
                    span_ends.append(end)

            ind += 1
            metadata['turn_id'] = ind

            instance = self._dataset_reader.text_to_instance(question_text,
                                                             paragraph,
                                                             zip(span_starts, span_ends),
                                                             answer_texts,
                                                             yesno,
                                                             tokenized_paragraph,
                                                             metadata)
            # yesno dealing stop here 18.10.29
            return instance

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs["id"] = instance.fields["metadata"].metadata["id"]
        outputs["turn_id"] = instance.fields["metadata"].metadata["turn_id"]
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for instance, output in zip(instances, outputs):
            output["turn_id"] = instance.fields["metadata"].metadata["turn_id"]
            output["id"] = instance.fields["metadata"].metadata["id"]
        return sanitize(outputs)
