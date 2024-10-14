# Step 1: Import Libraries and Load the Model
import numpy as np
# import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('rnn_imdb.h5')


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Streamlit app
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    processed_input = preprocess_text(user_input)

    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Dislay the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write("Please enter a movie review.")
