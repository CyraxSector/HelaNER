import argparse
import os

from model.tagger import Tagger
from model.models import load_model
from model.preprocessing import IndexTransformer
from mapper.ontoLoader import ontoLoader


def main(args):
    model_path = 'E:\\Projects\\HelaNER\\data\\hela-ontology.owl'
    mapper = ontoLoader()
    mapper.load_kb(model_path)

    print('Loading objects...')
    model = load_model(args.weights_file, args.params_file)
    it = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=it)

    print('Tagging a sentence...')
    res = tagger.analyze(args.sent)

    entities = []
    for entity in res['entities']:
        for canonical_name in mapper.load_kb(model_path).entities:
            if canonical_name.canonical_name.lstrip('#') == entity['text']:
                entities.append(entity['text'])
                entities.append(" ")


if __name__ == "__main__":
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'E:\\Projects\\HelaNER\\demo\\models')
    parser = argparse.ArgumentParser(description='Tagging a sentence.')
    parser.add_argument('--sent', default='තමා හයිටිය අතහැර පළා ගියේ කැමැත්තකින්.')
    parser.add_argument('--save_dir', default=SAVE_DIR)
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'E:\\Projects\\HelaNER\\demo\\models'
                                                                         '\\weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'E:\\Projects\\HelaNER\\demo\\models\\params'
                                                                        '.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'E:\\Projects\\HelaNER\\demo\\models'
                                                                              '\\preprocessor.pickle'))
    args = parser.parse_args()
    main(args)
