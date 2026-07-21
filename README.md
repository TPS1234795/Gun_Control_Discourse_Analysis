# 🔫 Gun Control Discourse Analysis

An AI-powered sentiment analysis dashboard that analyzes Reddit discussions on gun control using TF-IDF, Linear SVM, Streamlit, and Plotly.

---

## 📌 Project Overview

This project classifies Reddit comments into:

- 😊 Positive
- 😐 Neutral
- 😠 Negative

The dashboard is built with Streamlit and provides interactive visualizations, sentiment prediction, and prediction history.

---

## 🚀 Features

- 📊 Interactive Dashboard
- 😊 Sentiment Prediction
- 📈 Dataset Statistics
- 🥧 Sentiment Distribution Charts
- 🔑 Top Words by Sentiment
- 🎯 Model Performance Metrics
- 📜 Prediction History
- 📥 Download Predictions as CSV
- 🎨 Professional Dark Theme

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorizer
- Linear SVM
- Plotly
- Pandas
- NumPy

---

## 📂 Project Structure

```text
Gun_Control_Discourse_Analysis/
│
├── app.py
├── charts.py
├── utils.py
├── requirements.txt
│
├── assets/
│   ├── banner.png
│   ├── logo.png
│   └── style.css
│
├── data/
│   └── Reddit_Data.csv
│
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
```

---

## 📊 Dataset

- Source: Reddit Discussions
- Total Comments: 37,149
- Classes:
  - Positive
  - Neutral
  - Negative

---

## 🤖 Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. TF-IDF Feature Extraction
4. Linear SVM Training
5. Sentiment Prediction
6. Interactive Dashboard

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | **84.41%** |
| F1 Score | **0.84** |
| Model | Linear SVM |

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/TPS1234795/Gun_Control_Discourse_Analysis.git
```

Go to the project directory:

```bash
cd Gun_Control_Discourse_Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Dashboard

> Add screenshots of your dashboard here after deployment.

---

## 👨‍💻 Author

**Taniprava Sahoo**

B.Tech CSE (AI & ML)

GitHub:
https://github.com/TPS1234795

---

## ⭐ Future Improvements

- Deep Learning Models (LSTM, BERT)
- Live Reddit API Integration
- User Authentication
- Multi-language Support
- Explainable AI (SHAP/LIME)
- Real-time Analytics

---

## 📄 License

This project is licensed under the MIT License.
