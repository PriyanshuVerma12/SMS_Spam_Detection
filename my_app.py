import sklearn
import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')

model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
st.title('Email/SMS Spam Detector')


message = st.text_area('Enter Your Message')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Detect'):
    t_message = transform_text(message)

    vec_message = tfidf.transform([t_message])

    prediction = model.predict(vec_message)[0]

    if prediction == 0:
        st.header('Not Spam')
    else:
        st.header('Spam')


