import argparse
import os

from flask import Flask, render_template, request
from model.tagger import Tagger
from model.models import load_model
from model.preprocessing import IndexTransformer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=["POST"])
def web_runner():
    if request.method == 'POST':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        inputSentence = request.form['sentence']
        SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
        parser = argparse.ArgumentParser(description='Tagging a sentence.')
        #parser.add_argument('--sent', default='වාසුදේවට අහුඋනොත් ඔක්කොම ගුටි කනවා යකෝ.')
        parser.add_argument('--sent', default=inputSentence)
        parser.add_argument('--save_dir', default=SAVE_DIR)
        parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'weights.h5'))
        parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
        parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.pickle'))
        args = parser.parse_args()

        model = load_model(args.weights_file, args.params_file)
        it = IndexTransformer.load(args.preprocessor_file)
        tagger = Tagger(model, preprocessor=it)

        print('Tagging a sentence...')
        result = tagger.analyze(args.sent)
        return render_template('result.html', resultwords=result['words'], resultentities=result['entities'])


if __name__ == '__main__':
    app.run()
