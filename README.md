# 🎬 MovieSentix — Movie Review Sentiment Analyzer with RNN

## Table of Contents
- [Demo](#demo)
- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Tech Stack](#tech-stack)
- [Deployment on Streamlit](#deployment-on-streamlit)
- [Project Structure](#project-structure)
- [Bug / Feature Request](#bug--feature-request)
- [Future Scope](#future-scope)
- [Technology Used](#technology-used)
- [Author](#author)
---

## Demo

[🚀 **Live App** – Click here to test the model! Curious what it thinks? Enter your review to see its sentiment prediction.](https://moviesentimentanalysisusingrnn-ekfjnig9vjna7xforappdyn.streamlit.app/)


📷 **Screenshots**: _Add screenshots of the UI here_

---

## Overview

**MovieSentix** is an AI-powered sentiment analysis system that uses a Recurrent Neural Network (RNN) to classify movie reviews as **positive** or **negative**. Built with Python, TensorFlow/Keras, and Streamlit, the application provides accurate, real-time predictions through an intuitive web interface. It assists film enthusiasts, marketers, and data scientists in understanding audience sentiment from textual reviews.

---

## Motivation

In the era of digital reviews, understanding the sentiment behind thousands of user comments can be challenging. Manual review is time-consuming and inconsistent. MovieSentix was created to automate this task using natural language processing and deep learning. It empowers users with quick, reliable sentiment analysis—enabling smarter content decisions and feedback interpretation.

---

## Features

- 📊 **Real-Time Sentiment Prediction**: Instantly classifies reviews as *positive* or *negative* using a trained RNN model.
- 🧠 **Deep Learning Powered**: Built with an RNN using TensorFlow/Keras and trained on IMDB review datasets.
- 🎯 **High Accuracy**: Achieved >85% accuracy on test data through fine-tuned model training.
- 💬 **Interactive Interface**: Easy-to-use Streamlit web app for live text input and prediction.
- 📁 **Preprocessed Data Handling**: Supports batch inference on uploaded review files.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shrimanthv/Movie_Sentiment_Analysis_using_RNN.git
cd Movie_Sentiment_Analysis_using_RNN
```

### 2. Create and Activate a Virtual Environment
Using conda:

```bash
conda create -p sentiment-env python=3.10
conda activate sentiment-env
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
streamlit run app.py
```
### Tech Stack
| Layer      | Tools Used                                     |
| ---------- | ---------------------------------------------- |
| Frontend   | Streamlit                                      |
| Backend    | Python, TensorFlow/Keras, NumPy                |
| Model      | RNN for Sentiment Classification               |
| Data       | IMDB Movie Review Dataset                      |
| Deployment | Local (extendable to Streamlit Cloud / Heroku) |

### Deployment on Streamlit
1. To deploy the app on Streamlit Cloud:

2. Log in or sign up at Streamlit Cloud.

3. Connect your GitHub repository.

4. Select the repo and app.py as the entry file.

5. Make sure requirements.txt is included.

6. Click Deploy.

### Project Structure
│
├── app.py                  # Main Streamlit application
├── model.h5                # Trained RNN model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── utils/                  # Helper functions (e.g., tokenizer, preprocessing)
│   └── preprocess.py
├── data/                   # Raw and processed datasets
│   └── imdb_reviews.csv
└── notebooks/              # Jupyter notebooks for EDA & model training
    └── model_training.ipynb

## Bug / Feature Request
If you encounter any bugs or have suggestions for new features, please feel free to open an issue on the GitHub repository.

To report a bug:
Provide a clear description of the problem, steps to reproduce it, and any relevant screenshots or error messages.

To request a feature:
Describe the new functionality you'd like to see and explain how it would improve the project.

Your feedback helps improve AeroFare — thank you for contributing!

###  Future Scope
🧾 Add support for multi-class sentiment (e.g., neutral, mixed)

📉 Visualize review trends with sentiment over time

🌍 Multilingual sentiment support

📦 Deploy via Docker or Streamlit Cloud for broader accessibility

### Technology Used
<p align="center">
  <img src="https://www.python.org/static/community_logos/python-logo.png" width="140" alt="Python Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" width="120" alt="TensorFlow Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Keras_logo.svg" width="100" alt="Keras Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg" width="160" alt="Streamlit Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://pandas.pydata.org/static/img/pandas_mark.svg" width="100" alt="Pandas Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://numpy.org/images/logo.svg" width="120" alt="NumPy Logo" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://jupyter.org/assets/homepage/main-logo.svg" width="100" alt="Jupyter Logo" />
</p>

## Author
Shrimanth V
Email: shrimanthv99@gmail.com
Feel free to reach out for any questions or collaboration!