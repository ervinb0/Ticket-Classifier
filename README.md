# 🏷️ Jira Ticket Classifier

A machine learning web app that classifies Jira tickets into **Bug**, **Enhancement**, or **Question**, using the ticket’s title and description.

## 🚀 Features

- Interactive web interface using Flask + HTML/CSS/JS
- Real-time prediction from trained Random Forest model
- Useful for support teams, triage automation, or dashboards

## 🧠 Tech Stack

- Python (Flask, scikit-learn)
- HTML / CSS / JavaScript
- pandas, nltk for preprocessing

## 🌐 Web App Usage

1. Install dependencies:
     pip install -r requirements.txt
2. Run the app:
     python app.py
3. Open your browser and go to:
     http://127.0.0.1:5000
4. Enter the Jira ticket title and description to get a prediction.


## ✅ Todo

Deploy to Render/Heroku
Add support for bulk predictions via file upload
