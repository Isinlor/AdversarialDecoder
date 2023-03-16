from typing import List, Tuple

import numpy
from transformers import Pipeline

class IsFakePipeline():

    def predict(self, texts) -> List[float]:
        pass

    def getLeastFake(self, texts: List[str]) -> Tuple[List[str], float]:
        classifications = numpy.array(self.predict(texts))
        return texts[classifications.argmin()], classifications.min(initial=1)


class IsFakePipelineHF(Pipeline):

    def predict(self, texts):
        outputs = super().__call__(texts)
        scores = numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        return numpy.array([item[0] for item in scores])

    def getLeastFake(self, texts: List[str]) -> Tuple[List[str], float]:
        classifications = numpy.array(self.predict(texts))
        return texts[classifications.argmin()], classifications.min(initial=1)

class IsFakePipelineSklearn(IsFakePipeline):

    def __init__(self, model, vectorizer):
        super().__init__()
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, texts):
        # if texts is not list:
        #     texts = [texts]
        scores = self.model.predict_proba(self.vectorizer.transform(texts))
        return numpy.array([item[1] for item in scores])