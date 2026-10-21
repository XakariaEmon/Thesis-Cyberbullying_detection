# Multilingual Cyberbullying Detection Using Transformer Ensembles and Explainable AI

A multilingual NLP project for detecting and classifying cyberbullying in **English, Bangla, and mixed-language text** using transformer-based models, weighted soft voting, and Explainable AI (XAI).

## Overview

Cyberbullying on social media can take different forms and is often difficult to detect automatically, especially when content involves multiple languages, slang, and culturally specific expressions. These challenges are more pronounced for low-resource languages such as Bangla.

This project explores whether combining multiple transformer models can improve cyberbullying classification across different linguistic settings. Nine transformer models were fine-tuned and evaluated individually, then combined through an accuracy-weighted soft-voting ensemble. LIME was used to interpret model predictions, and a Streamlit application was developed to demonstrate real-time classification and explanations.

## Research Objectives

- Improve cyberbullying detection using an ensemble of transformer models.
- Combine publicly available English and Bangla datasets into a unified classification framework.
- Compare individual models and the ensemble across English, Bangla, and multilingual datasets.
- Investigate the effectiveness of ensemble learning on mixed-language text.
- Apply Explainable AI to make model predictions easier to interpret.

## Key Features

- **Multilingual classification:** Supports English, Bangla, and mixed-language text.
- **Nine transformer models:** Evaluates different pretrained architectures for cyberbullying detection.
- **Weighted soft voting:** Combines class probabilities from multiple models using accuracy-based weights.
- **Multi-class prediction:** Classifies text into four categories: `Not Bully`, `Religious`, `Sexual`, and `Others`.
- **Explainable AI:** Uses LIME to highlight words contributing to individual predictions.
- **Real-time demonstration:** Provides an interactive Streamlit interface for text classification and explanation.

## Models Used

The study evaluates nine pretrained transformer models.

| Language setting | Models |
|---|---|
| English | BERT, RoBERTa, ALBERT, GPT-2, XLNet |
| Bangla | mBERT, BanglaBERT, XLM-R |
| Multilingual | mBERT, DistilBERT, XLM-R |

The model groups reflect the experiments conducted for each dataset. Some models, such as mBERT and XLM-R, are evaluated in more than one setting.

## Methodology

The project follows a pipeline from dataset preparation to model evaluation and explainability.

1. **Dataset collection:** Combine publicly available cyberbullying datasets for English and Bangla and prepare a multilingual dataset containing English, Bangla, and mixed-language text.
2. **Preprocessing and tokenization:** Process text using each model's corresponding tokenizer, with padding and truncation to a maximum sequence length of 128 tokens.
3. **Model fine-tuning:** Fine-tune the selected pretrained transformer models for multi-class text classification.
4. **Performance evaluation:** Evaluate individual models using accuracy, precision, recall, F1-score, Cohen's Kappa, Matthews Correlation Coefficient (MCC), and ROC-AUC.
5. **Ensemble development:** Combine the class-probability distributions of selected models through accuracy-weighted soft voting.
6. **Explainability:** Apply LIME to examine which words contribute to a model's predictions.
7. **Application development:** Integrate classification and explanations into a Streamlit-based demonstration.

### Weighted Soft-Voting Ensemble

Each model produces a probability distribution over the target classes. The distributions are combined using normalized accuracy-based weights.

For model \(m_i\), its weight is defined as:

\[
w_i=\frac{\operatorname{Accuracy}(m_i)}
{\sum_{j=1}^{n}\operatorname{Accuracy}(m_j)}
\]

The ensemble probability for an input \(x\) is:

\[
P_{\mathrm{ensemble}}(x)
=\sum_{i=1}^{n} w_i P_i(x)
\]

The predicted class is the class with the highest aggregated probability.

This approach gives greater influence to models with higher measured accuracy while retaining information from the other models.

## Results

The weighted voting ensemble outperformed the individual models on all three reported dataset settings.

### Best Results by Dataset

| Dataset | Best individual model | Individual accuracy | Ensemble accuracy | Ensemble F1-score |
|---|---|---:|---:|---:|
| English | BERT | 86.73% | **87.26%** | **87.17%** |
| Bangla | BanglaBERT | 85.97% | **87.64%** | **87.63%** |
| Multilingual | DistilBERT | 85.97% | **86.68%** | **86.68%** |

### Ensemble Performance

| Metric | English | Bangla | Multilingual |
|---|---:|---:|---:|
| Accuracy | 87.26% | **87.64%** | 86.68% |
| Precision | 87.10% | **87.68%** | 86.68% |
| Recall | 87.26% | **87.64%** | 86.67% |
| F1-score | 87.17% | **87.63%** | 86.68% |
| Cohen's Kappa | 82.65% | **83.39%** | 82.23% |
| MCC | 82.66% | **83.41%** | 82.24% |
| ROC-AUC | 97.95% | **97.33%** | 97.65% |

*Results are reported from the experiments described in the thesis.*

### Key Findings

- The ensemble improved accuracy over the strongest individual model by approximately **0.53 percentage points** on English, **1.67 percentage points** on Bangla, and **0.71 percentage points** on the multilingual dataset.
- The Bangla ensemble achieved the highest accuracy and F1-score among the three evaluated settings.
- Religious cyberbullying was generally classified more reliably than the more ambiguous categories.
- Confusion between `Not Bully` and `Others` remained a challenge, indicating opportunities for improved annotation quality and class discrimination.

## Explainable AI

LIME (Local Interpretable Model-agnostic Explanations) was incorporated to provide insight into individual model predictions. By highlighting text features that influence a prediction, the system helps users inspect why a particular category may have been assigned.

Explainability is particularly relevant to cyberbullying detection, where context, language, and cultural interpretation can affect how a message is classified. These explanations are intended to support human interpretation rather than serve as definitive evidence of a model's reasoning.

## Evaluation Metrics

The experiments use multiple metrics to assess overall performance and class discrimination:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positives that are correct.
- **Recall:** Proportion of actual positives identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Cohen's Kappa:** Agreement between predictions and ground-truth labels adjusted for chance.
- **MCC:** Balanced measure of classification quality based on true and false predictions.
- **ROC-AUC:** Measures how well the model distinguishes between classes.

Confusion matrices and class-wise classification reports are also used to examine model strengths and common misclassifications.

## Technology Stack

- **Language:** Python
- **Deep learning:** PyTorch, TensorFlow
- **NLP and model training:** Hugging Face Transformers, Hugging Face Trainer API
- **Data processing:** Custom PyTorch Dataset objects and model-specific tokenizers
- **Explainability:** LIME
- **Interactive application:** Streamlit
- **Visualization and evaluation:** Classification reports, confusion matrices, and ROC curves

*The implementation uses PyTorch-based training components; TensorFlow is listed among the project tools.*

## Project Structure

The following is a suggested organization for publishing the project on GitHub. Adjust it to match the files actually included in your repository.

```text
.
├── datasets/                 # Dataset files or preparation scripts
├── models/                   # Fine-tuned model artifacts, if included
├── notebooks/                # Training and evaluation notebooks
├── figures/                  # Performance plots and confusion matrices
├── app.py                    # Streamlit application, if included
├── requirements.txt          # Python dependencies
└── README.md
```

## Getting Started

The exact setup commands depend on the scripts and files included in the repository.

### 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the project's training and application scripts.

### 4. Run the Application

If the repository includes `app.py` and its required model artifacts:

```bash
streamlit run app.py
```

Dataset preparation, model training, and inference commands should be added here once the corresponding scripts and their arguments are confirmed.

## Limitations and Future Work

Potential directions for further development include:

- Improving the distinction between `Not Bully` and `Others`.
- Reviewing ambiguous or potentially mislabeled samples.
- Expanding coverage of Bangla slang, code-switching, and regional language variations.
- Evaluating performance on independent datasets and real-world social media content.
- Exploring more efficient models for deployment under limited computational resources.
- Improving the interpretability and reliability of explanations.

## Responsible Use

Automated cyberbullying detection can produce false positives and false negatives. Predictions should not be treated as definitive judgments about a person or message. Human review, careful handling of sensitive text, and consideration of cultural and linguistic context are important before taking action based on a model's output.

## Acknowledgments

This project was developed as an undergraduate thesis exploring transformer-based ensemble learning and explainable AI for multilingual cyberbullying detection.

## Citation

If you use or refer to this work, please cite the thesis:

**Title:** *Ensemble Learning in Cyberbullying Detection Using Transformer Models: A Multilingual and Explainable-AI Approach*

**Author:** Faria Islam

**Institution:** Military Institute of Science and Technology (MIST), Dhaka, Bangladesh

**Year:** 2025

---

**Keywords:** Cyberbullying Detection · Natural Language Processing · Transformer Models · Ensemble Learning · Multilingual NLP · Bangla NLP · Explainable AI · LIME · Text Classification
