import os
import re
import pickle
import pandas as pd 
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = 'spam_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'

MAX_WORDS = 5000
MAX_LEN = 50

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model():
    st.info("Training model... This might take a moment.")
    print("Training dataset...")
    
    # Check if dataset exists before reading
    if not os.path.exists('spam.csv'):
        st.error("Error: 'spam.csv' not found! Please place it in the same directory.")
        return False
        
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    print(df.head())
    print(df["label"].value_counts())
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["message"] = df["text"].apply(clean_text)
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["message"])

    sequences = tokenizer.texts_to_sequences(df["message"])
    x = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
    y = df["label"].values
    print(x.shape)
    print(y.shape)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=32))
    model.add(SimpleRNN(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    model.save(MODEL_PATH)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    predictions = (model.predict(X_test) > 0.5).astype("int")
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    st.success("Training complete!")
    return True

def predict_sms(message):
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    message = clean_text(message)
    sequence = tokenizer.texts_to_sequences([message])
    sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")
    probability = model.predict(sequence, verbose=0)[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability

# Train model automatically if it doesn't exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    train_model()

# --- Streamlit UI ---
st.title("SMS SPAM DETECTOR")
st.write("many to one RNN Example")
message=st.text_area("Enter SMS Message:")
if st.button("predict"):
    prediction,probability=predict_sms(message)
    st.success(prediction)
    st.write(
        "confidence:",
        round(probability*100,2),
        "%"
    )