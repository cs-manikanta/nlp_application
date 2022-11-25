import pandas as pd
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow.keras.layers as L
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
import cleantext
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import streamlit as st
import cv2
import sys
from deepface import DeepFace  
import speech_recognition as srg 
global emoticon
emoticon = {"joy":"😄","sad":"🙁","sadness":"🙁","fear":"😨","anger":"😡","love":"🥰","angry":"😡","surprise":"😮","neutral":"😐","disgust":"😖"}
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

file = open("C:/Users/KP13/Downloads/360-NLP-Project/Pages/tokenizer.pickle",'rb')
tokenizer = pickle.load(file)
file.close()

file = open("C:/Users/KP13/Downloads/360-NLP-Project/Pages/labelEncoder.pickle",'rb')
le = pickle.load(file)
file.close()

model = keras.models.load_model("C:/Users/KP13/Downloads/360-NLP-Project/Pages/Emotion Recognition.h5")
def emotion(sentence):
    sentence = clean(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=256, truncating='pre')
    result = le.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  round(float(np.max(model.predict(sentence))),2)
    output = f"{result} {emoticon[result]}: {proba}"
    return output

st.header('Sentiment Analysis')
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        st.write(analyze(score_vader(text)),score_vader(text))
        if len(text)>40:
            st.write(emotion(text))




st.header("Audio Emotion Recognition")
clicked = st.button("Start mic")




def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def emotion_audio(path):
    length = librosa.get_duration(filename=path)
    if length<5:
        data,sr=librosa.load(path,duration=2.5,offset=0.6)
        data, index = librosa.effects.trim(data)
        aud=extract_features(data,sr)
        audio=np.array(aud)
        test = np.zeros(2376)
        if len(test)>len(audio):
            test[:len(audio)]=audio
        else:
            test = audio[:len(test)]
        file = open("C:/Users/KP13/Downloads/360-NLP-Project/Pages/scaler.pickle",'rb')
        scaler = pickle.load(file)
        file.close()  
        test=scaler.transform(test.reshape(1,-1))
        test=np.expand_dims(test,axis=2)
        model = keras.models.load_model("C:/Users/KP13/Downloads/360-NLP-Project/Pages/audio_model.h5")
        y_pred = model.predict(test)
        y_pred = np.argmax(y_pred)
        sentiment_dict = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
        return(sentiment_dict[y_pred])
    else:
        r = srg.Recognizer()
        with srg.AudioFile(path) as source:
            # listen for the data (load audio to memory)
            audio_data = r.record(source)
            # recognize (convert from speech to text)
            text = r.recognize_google(audio_data)
            return(str(score_vader(text))+str(analyze(score_vader(text))))
# Sampling frequency
freq = 22050
  
# Recording duration
duration = 4
  
# Start recorder with the given values 
# of duration and sample frequency
if clicked:
    recording = sd.rec(int(duration * freq), 
                    samplerate=freq, channels=1) 
    # Record audio for the given number of seconds
    sd.wait()
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("C:/Users/araj18/Downloads/recording0.wav", freq, recording)
    path = "C:/Users/araj18/Downloads/recording0.wav"
    st.write(emotion_audio(path),emoticon[emotion_audio(path)])
    os.remove(path)

with st.expander('Analyze file'):
    file = st.file_uploader("File upload", type=["wav"])
    if file:
        st.write("Filename: ",file.name)

        with open(os.path.join("C:/Users/KP13/Downloads",file.name),"wb") as f:
            f.write(file.getbuffer())
            path = "C:/Users/KP13/Downloads/"+file.name
            st.write(emotion_audio(path),emoticon[emotion_audio(path)])
        os.remove(path)



  
def extractSentiment(pathIn):
    sentiment ={'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad':0, 'surprise': 0, 'neutral': 0}
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000))    # added this line 
        success,image = vidcap.read()
        if not success:
            continue
#         print ('Read a new frame: ', success)
        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray,1.1,4)
        res = type(faces) is tuple
        if res==False:
            try:
                result = DeepFace.analyze(image,actions=['emotion'])
                for i in sentiment.keys():
                    sentiment[i] += result['emotion'][i]
#                 print(result)
            except:
                pass
#         cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1
    y = [round(i*100/sum(list(sentiment.values())),2) for i in list(sentiment.values())]
    plt.bar(range(len(sentiment)),y,color = ['red','green','purple','yellow','brown','blue','pink'])
    plt.xticks(range(len(sentiment)),sentiment.keys())
    st.pyplot(plt)
#     print(count)
#     print(sentiment)
#     print(max(sentiment,key = sentiment.get))
    s = max(sentiment,key = sentiment.get)
    output = s+emoticon[s]
    return output



st.header("Video Emotion Recognition")
c_clicked = st.button("Turn on Camera")


with st.expander('Analyze file'):
    file = st.file_uploader("File upload", type=["mp4"])
    if file:
        st.write("Filename: ",file.name)

        with open(os.path.join("C:/Users/KP13/Downloads",file.name),"wb") as f:
            f.write(file.getbuffer())
            path = "C:/Users/KP13/Downloads/"+file.name
            st.write(extractSentiment(path))
        os.remove(path)
        # st.write("Filename: ",file.name)
        # with tempfile.NamedTemporaryFile(mode="wb") as temp:
        #     bytes_data = file.getvalue()
        #     temp.write(bytes_data)
        #     print(temp.name)
            # st.write(extractSentiment(temp.name))


if c_clicked:
    st.write("Enter q to quit")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 2)
    # cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))  
    while(True):  
        # Capture image frame-by-frame  
        ret, frame = cap.read()  
      
        # Our operations on the frame come here  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
        # Display the resulting frame  
          
        faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces=faceCascade.detectMultiScale(gray,1.1,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        res = type(faces) is tuple
        if res==False:
            print("Face Detected")
            try:
                result = DeepFace.analyze(frame,actions=['emotion'])
                print(result['dominant_emotion'])
                cv2.putText(img = frame,text = result['dominant_emotion'],org = (100, 100),fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),thickness = 3)
            except:
                pass
        cv2.imshow('frame',frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()
    



