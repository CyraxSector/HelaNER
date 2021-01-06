import re
import string
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bs4 import BeautifulSoup
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense, LSTM, Activation, Input, Embedding
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from unidecode import unidecode

trainingDataPath = "https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/MScHateSpeechRefined.csv"
testDataPath = "https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/TestSpeechPhrases.csv"
pronounsListPath = "https://s3-ap-southeast-1.amazonaws.com/hatespeechcorpusmsc/Pronouns.csv"
trainingData_df = pd.read_csv(trainingDataPath, usecols=[0, 1, 2], header='infer', encoding='utf-8')
pronouns_df = pd.read_csv(pronounsListPath, usecols=[0], header=None, encoding='utf-8')
pronounsList = pronouns_df[0].values.tolist()


def RemoveHTMLElementsForDf(inputListDf):
    try:
        for i in range(len(inputListDf)):
            currentPhase = inputListDf['Phrase'].values[i]
            inputListDf['Phrase'].values[i] = BeautifulSoup(currentPhase, "html.parser").get_text()
        return inputListDf
    except Exception as e:
        e = sys.exc_info()[1]
        fullError = "RemoveHTMLElementsForDf: " + str(e) + "\n"
        raise Exception(fullError)


def ConvertToUnideCodeForDf(phraseDf):
    try:
        phraseEnglishSyllableWord = []
        phraseEnglishSyllableSentence = []

        for i in range(len(phraseDf)):
            currentPhrase = phraseDf['Phrase'].values[i]
            # split into words
            splitPhrase = currentPhrase.split()

            # Save each word with the unicode representation
            for splitPhraseElem in splitPhrase:
                phraseEnglishSyllableWord.append([splitPhraseElem, unidecode(splitPhraseElem)])

            # Then save the whole sentence with the unicode representation
            phraseEnglishSyllableSentence.append([currentPhrase, unidecode(currentPhrase)])

            # update the dataframe with the unidecode representation
            phraseDf['Phrase'].values[i] = unidecode(currentPhrase.lower())
        return phraseDf
    except:
        e = sys.exc_info()[1]
        print("ConvertToUnideCodeForList: " + str(e) + "\n")
        raise Exception


def RemoveSpecialCharactersForDf(inputListDf, pronounsList):
    try:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        cleanedList = []

        for i in range(len(inputListDf)):
            currentPhrase = inputListDf['Phrase'].values[i]
            tokenizedList = []
            punctuationSplittedList = []
            punctuationRemovedString = ""
            for splitPhrase in currentPhrase.split():
                splitPhrase = re.sub('\@\w+', '', splitPhrase)
                splitPhrase = re.sub('\#\w+', '', splitPhrase)
                splitPhrase = re.sub('\#', '', splitPhrase)
                splitPhrase = re.sub('RT', '', splitPhrase)
                splitPhrase = re.sub('&amp;', '', splitPhrase)
                splitPhrase = re.sub('[0-9]+', '', splitPhrase)
                splitPhrase = re.sub('//t.co/\w+', '', splitPhrase)
                splitPhrase = re.sub('w//', '', splitPhrase)
                splitPhrase = splitPhrase.lower()
                tokenizedList.append(splitPhrase.split())

            for tokenizedElem in tokenizedList:
                punctuation_Removed_Elem = regex.sub('', str(tokenizedElem))
                punctuationSplittedList.append(punctuation_Removed_Elem)

            for elem in punctuationSplittedList:
                if elem not in pronounsList:
                    punctuationRemovedString += (" " + elem.lower())

            inputListDf['Phrase'].values[i] = punctuationRemovedString
        return inputListDf
    except Exception as e:
        e = sys.exc_info()[1]
        fullError = "RemoveSpecialCharactersForDf: " + str(e) + "\n"
        raise Exception(fullError)


def RemoveNonWordEntriesForDf(inputDf):
    try:
        for i in range(len(inputDf)):
            currentPhase = inputDf['Phrase'].values[i]
            # Then append back after checking for numerical values
            inputDf['Phrase'].values[i] = (
                ' '.join([word for word in currentPhase.split() if word.isdigit() is False]))
        return inputDf
    except Exception as e:
        e = sys.exc_info()[1]
        fullError = "RemoveNonWordEntriesForDf: " + str(e) + "\n"
        raise Exception(fullError)


# Operation 1 : Cleaning the data
htmlCleanedTraining_Df = RemoveHTMLElementsForDf(trainingData_df)
unideCode_Training_Df = ConvertToUnideCodeForDf(htmlCleanedTraining_Df)
specialSymbolCleanedPhrases_Df = RemoveSpecialCharactersForDf(unideCode_Training_Df, pronounsList)
speechCleaned_NonWord_Df = RemoveNonWordEntriesForDf(specialSymbolCleanedPhrases_Df)
speechCleaned_NonWord_Df['IsHateSpeech'] = speechCleaned_NonWord_Df['IsHateSpeech'].map({'YES': 1, 'NO': 0})
speechCleaned_NonWord_Df = speechCleaned_NonWord_Df.head(2000)
speechCleaned_NonWord_Df = shuffle(speechCleaned_NonWord_Df)

sns.countplot(speechCleaned_NonWord_Df.IsHateSpeech)
plt.xlabel('Label')
plt.title('Number of hate speech')

print(len(speechCleaned_NonWord_Df))

X = speechCleaned_NonWord_Df.Phrase
Y = speechCleaned_NonWord_Df.IsHateSpeech
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
max_words = 50
max_len_arr = [50, 100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
bench_mark_results = []

for max_len in max_len_arr:
    startTime = time.time()
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    train_sequences = tok.texts_to_sequences(X_train)
    train_sequences_matrix = sequence.pad_sequences(train_sequences, maxlen=max_len)


    # Define the RNN structure.
    def RNN():
        inputs = Input(name='inputs', shape=[max_len])
        layer = Embedding(max_words, 50, input_length=max_len)(inputs)
        layer = LSTM(10)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        return model


    model = RNN()
    model.compile(loss='mean_absolute_error', optimizer=RMSprop(), metrics=['accuracy'])

    model.fit(train_sequences_matrix, Y_train, batch_size=128, epochs=10, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    # Process the test set data.
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    # Evaluate the model on the train set and test set.
    score_train = model.evaluate(train_sequences_matrix, Y_train)
    score_test = model.evaluate(test_sequences_matrix, Y_test)

    val_predict_train = (np.asarray(model.predict(train_sequences_matrix))).round()
    val_predict_test = (np.asarray(model.predict(test_sequences_matrix))).round()
    precision_train = precision_score(Y_train, val_predict_train, average='micro')
    precision_test = precision_score(Y_test, val_predict_test, average='micro')
    recall_train = recall_score(Y_train, val_predict_train, average='micro')
    recall_test = recall_score(Y_test, val_predict_test, average='micro')
    f1_score_train = f1_score(Y_train, val_predict_train, average='micro')
    f1_score_test = f1_score(Y_test, val_predict_test, average='micro')

    # print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr_train[0],accr_train[1]))
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr_test[0],accr_test[1]))
    endTime = time.time()
    elapsedTime = endTime - startTime
    bench_mark_results.append({'max_len': max_len,
                               'Precision[Training](%)': str(round(precision_train, 2) * 100),
                               'F1-Score[Training](%)': str(round(f1_score_train, 2) * 100),
                               'Recall[Training](%)': str(round(recall_train, 2) * 100),
                               'Mean Absolute Error[Training](%)': str(round(score_train[0], 2) * 100),
                               'Precision[Test](%)': str(round(precision_test, 2) * 100),
                               'F1-Score[Test](%)': str(round(f1_score_test, 2) * 100),
                               'Recall[Test](%)': str(round(recall_test, 2) * 100),
                               'Mean Absolute Error[Test](%)': str(round(score_test[0], 2) * 100),
                               'Precision_Difference(%)': str((round((precision_train - precision_test), 2) * 100)),
                               'Elapsed_Time(Sec.)': str(round(elapsedTime, 2))})

print(bench_mark_results)
