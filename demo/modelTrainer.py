import argparse
import os
import numpy as np

from model.utils import load_data_and_labels
from model.models import BiLSTMCRF
from model.preprocessing import IndexTransformer
from model.trainer import Trainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(args):
    print('Loading dataset...')
    x_train, y_train = load_data_and_labels(args.train_data)
    x_valid, y_valid = load_data_and_labels(args.valid_data)
    x_test, y_test = load_data_and_labels(args.test_data)
    x_train = np.r_[x_train, x_valid]
    y_train = np.r_[y_train, y_valid]

    print('Transforming datasets...')
    p = IndexTransformer(use_char=args.no_char_feature)
    p.fit(x_train, y_train)

    print('Building a model.')
    model = BiLSTMCRF(char_embedding_dim=args.char_emb_size,
                      word_embedding_dim=args.word_emb_size,
                      char_lstm_size=args.char_lstm_units,
                      word_lstm_size=args.word_lstm_units,
                      char_vocab_size=p.char_vocab_size,
                      word_vocab_size=p.word_vocab_size,
                      num_labels=p.label_size,
                      dropout=args.dropout,
                      use_char=args.no_char_feature,
                      use_crf=args.no_use_crf)
    model, loss = model.build()
    model.compile(loss=loss, optimizer='adam')

    print('Training the model...')
    trainer = Trainer(model, preprocessor=p)
    trainer.train(x_train, y_train, x_test, y_test)

    print('Saving the model...')
    model.save(args.weights_file, args.params_file)
    p.save(args.preprocessor_file)


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/conll2003/hela/ner')
    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--train_data', default=os.path.join(DATA_DIR, 'train.txt'), help='training data')
    parser.add_argument('--valid_data', default=os.path.join(DATA_DIR, 'valid.txt'), help='validation data')
    parser.add_argument('--test_data', default=os.path.join(DATA_DIR, 'test.txt'), help='test data')
    parser.add_argument('--weights_file', default='weights.h5', help='weights file')
    parser.add_argument('--params_file', default='params.json', help='parameter file')
    parser.add_argument('--preprocessor_file', default='preprocessor.pickle', help='preprocessor file')
    # Training parameters
    parser.add_argument('--loss', default='categorical_crossentropy', help='loss')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--max_epoch', type=int, default=15, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--checkpoint_path', default=None, help='checkpoint path')
    parser.add_argument('--log_dir', default=None, help='log directory')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping')
    # Model parameters
    parser.add_argument('--char_emb_size', type=int, default=25, help='character embedding size')
    parser.add_argument('--word_emb_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--char_lstm_units', type=int, default=25, help='num of character lstm units')
    parser.add_argument('--word_lstm_units', type=int, default=100, help='num of word lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--no_char_feature', action='store_false', help='use char feature')
    parser.add_argument('--no_use_crf', action='store_false', help='use crf layer')

    args = parser.parse_args()
    main(args)
