# 📩 SMS Spam Detection using Simple RNN

A **Many-to-One Recurrent Neural Network (RNN)** project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using TensorFlow/Keras. The project includes a user-friendly **Streamlit** web application for real-time predictions.

---

## 📌 Project Overview

Spam SMS messages are a common problem in mobile communication. This project applies a **Simple RNN** to learn sequential patterns in SMS text and accurately classify messages into two categories:

* ✅ Ham (Legitimate Message)
* 🚨 Spam (Unwanted Message)

The model is trained on the popular **SMS Spam Collection Dataset** (`spam.csv`).

---

## 🚀 Features

* Text preprocessing and cleaning
* Tokenization using Keras Tokenizer
* Sequence padding
* Embedding Layer for word representation
* Simple RNN for sequence learning
* Binary classification using Sigmoid activation
* Model evaluation with:

  * Accuracy
  * Confusion Matrix
  * Classification Report
* Interactive Streamlit web application
* Automatic model and tokenizer saving

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Pickle

---

## 📂 Project Structure

```text
SMS_Spam_Detection/
│
├── app.py                 # Main Streamlit application
├── spam.csv               # SMS Spam Collection dataset
├── spam_model.keras       # Saved trained model (generated automatically)
├── tokenizer.pkl          # Saved tokenizer (generated automatically)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SMS-Spam-Detection-RNN.git
```

### 2. Navigate to the project directory

```bash
cd SMS-Spam-Detection-RNN
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

The application will automatically train the model if `spam_model.keras` is not found.

---

## 🧠 Model Architecture

```text
Input SMS
      │
      ▼
Text Cleaning
      │
      ▼
Tokenizer
      │
      ▼
Padding
      │
      ▼
Embedding Layer
      │
      ▼
Simple RNN Layer
      │
      ▼
Dense (ReLU)
      │
      ▼
Dense (Sigmoid)
      │
      ▼
Prediction
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score

---

## 💬 Example

**Input**

```text
Congratulations! You've won a FREE vacation. Call now to claim your prize.
```

**Prediction**

```text
🚨 Spam
Confidence: 98.75%
```

---

**Input**

```text
Hey, I'll reach home by 6 PM. See you soon!
```

**Prediction**

```text
✅ Ham
Confidence: 99.12%
```

---

## 📚 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages classified as **spam** or **ham**.

Dataset columns:

* `v1` → Label (`ham` or `spam`)
* `v2` → SMS Message

---

## 🎯 Future Improvements

* Replace Simple RNN with LSTM or GRU
* Hyperparameter tuning
* Attention mechanism
* Bidirectional RNN
* Deploy on Streamlit Community Cloud
* REST API using FastAPI or Flask
* Support for multiple languages

---

## 👨‍💻 Author

**Krishna Anand Chauhan**

B.Tech – Computer Science & Engineering (Data Science)

Passionate about Artificial Intelligence, Machine Learning, Deep Learning, and Data Science.

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star** on GitHub. It helps others discover the project and motivates further development.
