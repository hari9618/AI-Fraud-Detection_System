# 💳 AI-Powered Fraud Intelligence System

An end-to-end Machine Learning web application that detects fraudulent credit card transactions using a hybrid approach:

- 📊 Data Visualization Dashboard  
- 📈 Association Rule Mining (Apriori Algorithm)  
- 🤖 Random Forest Machine Learning Model  
- 🚀 Interactive Streamlit Web Application  

---

## 🚀 Live Demo

🔗 Streamlit App: https://ai-fraud-detectionsystem-dzmoujfapu57gk7hne8avv.streamlit.app/

🔗 GitHub Repository: 

---

## 📌 Project Overview

Credit card fraud is a major financial threat worldwide.  
This project builds a Hybrid AI System combining:

1. **Association Rule Mining** to discover fraud patterns
2. **Random Forest Classifier** for fraud prediction
3. **Interactive dashboard** for real-time analysis

The application is deployed using **Streamlit Cloud**.

---

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Mlxtend (Apriori Algorithm)

---

## 📂 Dataset

Dataset used:
- Credit Card Fraud Detection Dataset (Kaggle)

The dataset contains:
- 284,807 transactions
- 31 columns (V1–V28 anonymized features)
- `Amount`
- `Class` (0 = Legit, 1 = Fraud)

⚠ Note:
The dataset is NOT included in this repository due to size limitations.
Please download it from Kaggle and upload it when running the app.

---

## ⚙️ Features

### 📊 1. Dashboard
- Total transactions
- Fraud cases count
- Fraud rate percentage
- Transaction amount distribution

### 📈 2. Fraud Pattern Discovery
- Apriori algorithm
- Association rule mining
- Lift & confidence analysis
- Top fraud pattern visualization

### 🤖 3. Machine Learning Prediction
- Random Forest Classifier
- Stratified train-test split
- Fraud probability score (Risk Gauge)
- Custom transaction testing

---

## 🏗 Project Structure

```
AI-Fraud-Detection-System/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How To Run Locally

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/AI-Fraud-Detection-System.git
```

### 2️⃣ Install Requirements

```
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

### 4️⃣ Upload Dataset

Upload the `creditcard.csv` file when prompted.

---

## 🌍 Deployment

The application is deployed using **Streamlit Cloud**.

Deployment steps:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy `app.py`
4. Upload dataset when app runs

---

## 📈 Model Details

- Algorithm: Random Forest Classifier
- Estimators: 25
- Max Depth: 8
- Stratified Sampling
- Performance optimized for deployment

---

## 🎯 Key Highlights

✔ Hybrid AI approach  
✔ Real-time fraud risk scoring  
✔ Interactive UI  
✔ Optimized for cloud deployment  
✔ Production-style structure  

---

## 🔐 Important Note

- The dataset is large (~280MB) and not included in the repository.
- A smaller sampled dataset (50,000 rows) is recommended for deployment.
- Always ensure the dataset contains the `Class` column.

---

## 👨‍💻 Author

Thota Hari Krishna 
LinkedIn: https://www.linkedin.com/public-profile/settings/?trk=d_flagship3_profile_self_view_public_profile&lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BzlUYiWWNTMa0K6J65bGYWg%3D%3D 
GitHub: (Add your GitHub profile link)

---

## ⭐ If You Like This Project

Give this repository a ⭐ on GitHub!
