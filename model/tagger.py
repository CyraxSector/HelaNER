import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


class Tagger(object):
    def __init__(self, model, preprocessor, tokenizer=str.split):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

    def predict_proba(self, text):
        assert isinstance(text, str)

        words = self.tokenizer(text)
        X = self.preprocessor.transform([words])
        y = self.model.predict(X)
        y = y[0]

        return y

    def _get_prob(self, pred):
        prob = np.max(pred, -1)

        return prob

    def _get_tags(self, pred):
        tags = self.preprocessor.inverse_transform([pred])
        tags = tags[0]

        return tags

    def _build_response(self, sent, tags, prob):
        words = self.tokenizer(sent)
        res = {
            'words': words,
            'entities': [

            ]
        }
        chunks = get_entities(tags)

        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_end += 1
            entity = {
                'text': ' '.join(words[chunk_start: chunk_end]),
                'type': chunk_type,
                'score': float(np.average(prob[chunk_start: chunk_end])),
                'beginOffset': chunk_start,
                'endOffset': chunk_end
            }
            res['entities'].append(entity)

        return res

    def analyze(self, text):
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)
        prob = self._get_prob(pred)
        res = self._build_response(text, tags, prob)

        return res

    def predict(self, text):
        pred = self.predict_proba(text)
        tags = self._get_tags(pred)

        return tags
