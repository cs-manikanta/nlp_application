# # Core Pkgs
# import streamlit as st 
# import os

# # NLP Pkgs
# from textblob import TextBlob 
# import spacy

# # Sumy Summary Pkg
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer


# # Function for Sumy Summarization
# def sumy_summarizer(docx):
# 	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
# 	lex_summarizer = LexRankSummarizer()
# 	summary = lex_summarizer(parser.document,3)
# 	summary_list = [str(sentence) for sentence in summary]
# 	result = ' '.join(summary_list)
# 	return result

# # Function to Analyse Tokens and Lemma
# @st.cache
# def text_analyzer(my_text):
# 	nlp = spacy.load('en_core_web_sm')
# 	docx = nlp(my_text)
# 	# tokens = [ token.text for token in docx]
# 	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
# 	return allData

# # Function For Extracting Entities
# @st.cache
# def entity_analyzer(my_text):
# 	nlp = spacy.load('en_core_web_sm')
# 	docx = nlp(my_text)
# 	tokens = [ token.text for token in docx]
# 	entities = [(entity.text,entity.label_)for entity in docx.ents]
# 	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
# 	return allData

# def getAnalysis(score):
#     if score < 0:
#         return 'Negative'
#     elif score == 0:
#         return 'Neutral'
#     else:
#         return 'Positive'

# def main():

# 	# Title
# 	st.title("Text Analyzer with Streamlit")
# 	st.subheader("Natural Language Processing On the Go..")
# 	st.markdown("""
#     	#### Description
#     	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
#     	Tokenization,NER,Sentiment,Summarization
#     	""")

# 	# Tokenization
# 	if st.checkbox("Show Tokens and Lemma"):
# 		st.subheader("Tokenize Your Text")

# 		message = st.text_area("Enter Text","Type Here ..")
# 		if st.button("Analyze"):
# 			nlp_result = text_analyzer(message)
# 			st.json(nlp_result)

# 	# Entity Extraction
# 	if st.checkbox("Show Named Entities"):
# 		st.subheader("Analyze Your Text")

# 		message = st.text_area("Enter Text","Type Here ..")
# 		if st.button("Extract"):
# 			entity_result = entity_analyzer(message)
# 			st.json(entity_result)

# 	# Sentiment Analysis
# 	if st.checkbox("Show Sentiment Analysis"):
# 		st.subheader("Analyse Your Text")

# 		message = st.text_area("Enter Text","Type Here ..")
# 		if st.button("Analyze"):
# 			blob = TextBlob(message)
# 			result_sentiment = blob.sentiment
# 			st.success(getAnalysis(result_sentiment.polarity))

# 	# Summarization
# 	if st.checkbox("Show Text Summarization"):
# 		st.subheader("Summarize Your Text")

# 		message = st.text_area("Enter Text","Type Here ..")
# 		summary_options = st.selectbox("Choose Summarizer",['sumy'])
# 		if st.button("Summarize"):

# 			st.text("Using Sumy Summarizer ..")
# 			summary_result = sumy_summarizer(message)		
# 			st.success(summary_result)



# 	st.sidebar.subheader("About App")
# 	st.sidebar.text("Text Analyzer App with Streamlit")
	

	

# if __name__ == '__main__':
# 	main()


#########################
# Core Pkgs
import streamlit as st 
import os
import nltk 
nltk.download('punkt')
# NLP Pkgs
from textblob import TextBlob 
import spacy

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline #-------- Pooja----------#
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
#from nltk.tokenize import sent_tokenize

#Function for Sumy Summarization
def sum_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

from transformers import pipeline
def bart_summarizer(docx):
    summarizer = pipeline('summarization')#default model distilbart
    summary = summarizer(docx, max_length=170, min_length=50, do_sample=False)
    results = summary[0]['summary_text']
    return results
#---------------------------------Pooja-------------------------------------------------#
def pegasus(input_text):
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    batch = tokenizer(input_text, truncation=True, padding='longest', return_tensors="pt")
    #input_text=''''Mentally ill inmates are housed on the "forgotten floor" of a Miami jail .Judge Steven Leifman says the charges are usually "avoidable felonies"He says the arrests often result from confrontations with police .Mentally ill people often won\'t do what they\'re told when police arrive on the scene . '''
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text
#--------------------------------------Pooja---------------------------------------------#
# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

def main():

	# Title
	st.title("Text Analyzer with Streamlit")
	st.subheader("Natural Language Processing On the Go..")
	st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	Tokenization,NER,Sentiment,Summarization
    	""")

	# Tokenization
	if st.checkbox("Show Tokens and Lemma"):
		st.subheader("Tokenize Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		if st.button("Analyze"):
			nlp_result = text_analyzer(message)
			st.json(nlp_result)

	# Entity Extraction
	if st.checkbox("Show Named Entities"):
		st.subheader("Analyze Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		if st.button("Extract"):
			entity_result = entity_analyzer(message)
			st.json(entity_result)

	# Sentiment Analysis
	if st.checkbox("Show Sentiment Analysis"):
		st.subheader("Analyse Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		if st.button("Analyze"):
			blob = TextBlob(message)
			result_sentiment = blob.sentiment
			st.success(getAnalysis(result_sentiment.polarity))

	# Summarization
	if st.checkbox("Show Text Summarization"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter Text","Type Here ..")
		summary_options = st.selectbox("Choose Summarizer",['sumy','bart','pegasus'])
		if summary_options == 'bart' and st.button("Summarize"):

			st.text("Using bart Summarizer ..")
			summary_result = bart_summarizer(message)		
			st.success(summary_result)
	#-----------------------------------------Pooja---------------------------------------------------#
		elif summary_options == 'pegasus'and st.button("Summarize"):
			st.text("Using pegasus Summarizer..")
			summary_result = pegasus(message)
			st.success(summary_result)
	#----------------------------------------------Pooja-----------------------------------#	
		else:
			if summary_options == 'sumy' and st.button("Summarize"):
				st.text("Using Sumy Summarizer ..")
				summary_result = sum_summarizer(message)		
				st.success(summary_result)
			else:
				pass



	st.sidebar.subheader("About App")
	st.sidebar.text("Text Analyzer App with Streamlit")
	

	

if __name__ == '__main__':
	main()