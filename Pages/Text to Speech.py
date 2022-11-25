import os

import gtts as g
import streamlit as st

from gtts import gTTS
    

def main():
    st.title('Text to speech')
    text = st.text_input('Enter text')
    lang1 = st.selectbox('Select language',('English', 'Hindi', 'Spanish', 'German'))
    print(lang1)
    lang_dict={'English':'en', 'Spanish':'es','Hindi':'hi', 'German':'de'}
    lang=lang_dict.get(lang1)

    if st.button('Convert'):
        audio = gTTS(text=text, lang=lang, slow=False)
        
        audio.save('text-to-speech.mp3')
        st.audio('text-to-speech.mp3', format="audio/wav", start_time=0)
    

if __name__ == '__main__':
	main()










#audio1.save('C:\\Users\\vdhawas\\OneDrive - Capgemini\\Desktop\\speech_eng_eng.mp3')
#st.audio(audio1, format="audio/wav", start_time=0)
#st.audio('C:\\Users\\vdhawas\\OneDrive - Capgemini\\Desktop\\speech_eng_eng.mp3', format="audio/wav", start_time=0)

#audio.save("speech_eng_eng.mp3")

#os.system("start C:\\Users\\vdhawas\\OneDrive - Capgemini\\Desktop\\speech_eng_eng.mp3")




