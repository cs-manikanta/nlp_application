import streamlit as st
from transcribe import *
import time
import speech_recognition

st.header("Speech-To-Text App")


fileObject = st.file_uploader(label="Please upload your file")
if fileObject:
    token, t_id = upload_file(fileObject)
    result = {}
    sleep_duration = 1
    percent_complete = 0
    progress_bar = st.progress(percent_complete)

    st.text("Transcribing the file...")
    while result.get("status") != "processing":
        percent_complete += sleep_duration
        time.sleep(sleep_duration)
        progress_bar.progress(percent_complete / 10)
        result = get_text(token, t_id)

    sleep_duration = 0.01

    for percent in range(percent_complete, 101):
        time.sleep(sleep_duration)
        progress_bar.progress(percent)

    with st.spinner("Processing....."):
        while result.get("status") != 'completed':
            result = get_text(token, t_id)

    st.header("Transcribed Text")
    st.markdown(result['text'])

st.write('OR')

sr = speech_recognition.Recognizer()
with speech_recognition.Microphone() as source2:
    print(" Silence please")                                                                  
    sr.adjust_for_ambient_noise(source2, duration = 2)
    if st.button('Speak'):
        try:
            print(" Speak now please....")
            audio2 = sr.listen(source2)
            textt = sr.recognize_google(audio2)
            textt = textt.capitalize()
            st.write("Did you say :", textt)
        except:
            st.write("can you speak again?")
