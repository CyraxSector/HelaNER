from model.callbacks import F1score
from model.utils import NERSequence


class Trainer(object):
    def __init__(self, model, preprocessor=None):
        self._model = model
        self._preprocessor = preprocessor

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
              epochs=1, batch_size=32, verbose=1, callbacks=None, shuffle=True):
        train_seq = NERSequence(x_train, y_train, batch_size, self._preprocessor.transform)

        if x_valid and y_valid:
            valid_seq = NERSequence(x_valid, y_valid, batch_size, self._preprocessor.transform)
            f1 = F1score(valid_seq, preprocessor=self._preprocessor)
            callbacks = [f1] + callbacks if callbacks else [f1]

        self._model.fit_generator(generator=train_seq,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  verbose=verbose,
                                  shuffle=shuffle)
