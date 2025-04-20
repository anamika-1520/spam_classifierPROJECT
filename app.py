import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import nltk
import streamlit as st
# nltk.download('punkt')
import pickle

model=pickle.load(open('model.pkl','rb'))
tfid=pickle.load(open('tfid.pkl','rb'))

ps=PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
    
    return " ".join(y)

st.title('SMS Spam Classifier')

input_sms=st.text_area('Enter the message')

transformed_sms=transform_text(input_sms)
vector_input=tfid.transform([transformed_sms])
result=model.predict(vector_input)[0]

if st.button('Click to Know'):
    if result==1:
        st.header('Spam')
    else:
        st.header("Not Spam")