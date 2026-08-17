# Transparent AI System for Recruitment and Resume Screening (XAI)

An Explainable AI (XAI) solution for automated resume classification using **TF-IDF + Logistic Regression** and **LIME (Locally Interpretable Model-Agnostic Explanations)**. 

This project demonstrates how black-box recruiter screening algorithms can be made transparent, trustworthy, and auditable for keyword bias.

## Key Features
- **Automated Skill Classification:** Categorizes candidates into target job roles based on text content.
- **Local Explanations (LIME):** Highlights positive and negative word features that drove the decision.
- **Auditing Capability:** Allows HR managers to detect if unaligned keywords improperly sway the model.
- **Interactive Web Interface:** Built with Streamlit for real-time visual explanations.

## Tech Stack
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (TF-IDF Vectorizer, Logistic Regression)
- **Explainable AI Framework:** LIME (`lime_text`)
- **Web Dashboard:** Streamlit
- **Visualization:** Matplotlib

## Setup & Local Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/xai-resume-screening.git](https://github.com/your-username/xai-resume-screening.git)
   cd xai-resume-screening# Transparent-AI-for-Recruitment-and-Resume-Screening
