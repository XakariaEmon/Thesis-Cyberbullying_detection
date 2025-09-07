# 🎓 Ensemble Learning in Cyberbullying Detection Using Transformer Models: A Multilingual and Explainable AI Approach  

## 📖 Overview  
This repository contains the implementation of our undergraduate thesis project, where we developed a **multilingual cyberbullying detection system** using **transformer models** and an **ensemble learning approach**.  

Our research focuses on detecting harmful online content across **English** and **Bangla**, with an emphasis on explainability and real-time usability.  

---

## 👨‍🏫 Thesis Information  
- **Title:** Ensemble Learning in Cyberbullying Detection Using Transformer Models: A Multilingual and Explainable AI Approach  
- **Authors:**  
  - Ahmad Abdullah (ID: 202114008)  
  - Faria Islam (ID: 202114021)  
  - Md Zakaria Hossen Emon (ID: 202114060)  
- **Supervisor:** Prof. Dr. Fernaz Narin Nur, Dept. of CSE, MIST  
- **Institute:** Military Institute of Science and Technology (MIST), Dhaka, Bangladesh  
- **Conference Publication:** ✅ Published in 2025  

---

## 🧠 Motivation  
Cyberbullying is a rising concern on social media platforms. The challenge is amplified by anonymity, linguistic diversity, and the lack of effective tools for **low-resource languages like Bangla**.  
Our aim was to build an inclusive system that:  
- Detects **cyberbullying in both English & Bangla**  
- Provides **explainability** behind predictions  
- Works in **real-time**  

---

## ⚙️ Methodology  

### 1. Datasets  
- Collected and curated datasets from Kaggle and other sources  
- Created three datasets:  
  - English (~28k samples)  
  - Bangla (~24k samples)  
  - Multilingual (combined)  
- Labeled into 4 categories:  
  - 🕌 Religious  
  - 🔞 Sexual  
  - 💬 Others  
  - ✅ Not Bully  

### 2. Models Used  
- English: BERT, RoBERTa, ALBERT, GPT-2, XLNet  
- Bangla: mBERT, BanglaBERT, XLM-R  
- Multilingual: mBERT, DistilBERT, XLM-R  

### 3. Ensemble Learning  
- Applied a weighted soft-voting ensemble approach  
- Assigned higher weights to stronger models  
- Achieved better performance than standalone models  

### 4. Explainable AI  
- Used LIME (Local Interpretable Model-Agnostic Explanations)  
- Highlighted key words influencing model predictions  

### 5. Real-Time Interface  
- Built a Streamlit-based app for real-time detection  
- Users can input text and view predictions + confidence scores  

---

## 📊 Results  

| Dataset       | Best Accuracy | Best F1-score |
|---------------|--------------|---------------|
| English   | 87.26%       | 87.17%        |
| Bangla    | 87.64%       | 87.63%        |
| Multilingual | 86.68%    | 86.68%        |

- Ensemble consistently outperformed individual models  
- Strongest performance in religious class, weakest in sexual class due to overlap in language cues  

---

## 🚀 Features  
- Multilingual (English + Bangla) support  
- Transformer-based ensemble approach  
- Explainable AI with LIME  
- Real-time detection interface (Streamlit)  

---

## 🛠 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/XakariaEmon/cyberbullying-ensemble.git
   cd Thesis-Cyberbullying_detection
