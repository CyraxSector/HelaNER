import argparse
import os
from pprint import pprint

from model.tagger import Tagger
from model.models import load_model
from model.preprocessing import IndexTransformer


def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print('Loading objects...')
    model = load_model(args.weights_file, args.params_file)
    it = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=it)

    print('Tagging a sentence...')
    res = tagger.analyze(args.sent)
    pprint(res)


if __name__ == '__main__':
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    parser = argparse.ArgumentParser(description='Tagging a sentence.')
    parser.add_argument('--sent', default='රනිල් වික්‍රමසිංහ මහතා පැවසීය.')
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.pickle'))
    args = parser.parse_args()
    main(args)
