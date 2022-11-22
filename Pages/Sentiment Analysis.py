import pandas as pd
import streamlit as st
import cleantext
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras








def clean_text(text):
    return cleantext.clean(text, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)


def score_vader(text):
    text = GoogleTranslator(source='auto', target='en').translate(text)
    analyzer = SentimentIntensityAnalyzer()
    text = clean_text(text)
    vs = analyzer.polarity_scores(text)
    return vs['compound']


def analyze(x):
    if x > 0:
        return 'Positive'
    elif x < 0:
        return 'Negative'
    else:
        return 'Neutral'


def clean(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

file = open("C:/Users/araj18/Downloads/results (1)/tokenizer.pickle",'rb')
tokenizer = pickle.load(file)
file.close()

file = open("C:/Users/araj18/Downloads/results (1)/labelEncoder.pickle",'rb')
le = pickle.load(file)
file.close()

model = keras.models.load_model("C:/Users/araj18/Downloads/results (1)/Emotion Recognition.h5")
def emotion(sentence):
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    output = f"{result} : {proba}"
    return output

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        st.write(analyze(score_vader(text)),score_vader(text))
        if len(text)>40:
            st.write(emotion(text))







