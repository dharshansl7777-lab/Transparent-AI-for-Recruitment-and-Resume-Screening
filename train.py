import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1. Directory Setup
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 2. Data Loading / Synthetic Generation
data_path = "data/synthetic_resumes.csv"

if not os.path.exists(data_path):
    print("Generating synthetic resume dataset for training...")
    sample_data = {
        "resume_text": [
            "Senior Data Scientist with 6 years experience in Python, Machine Learning, TensorFlow, PyTorch, and SQL. Built predictive models and deployed pipelines.",
            "Data Analyst skilled in SQL, Tableau, PowerBI, Excel, and Python. Experienced in data visualization, dashboard creation, and ETL processes.",
            "Software Engineer with expertise in Java, Spring Boot, Microservices, Docker, Kubernetes, and REST APIs. Strong background in backend architecture.",
            "Frontend Developer proficient in React, JavaScript, HTML5, CSS, Tailwind, and Redux. Built responsive web applications and user interfaces.",
            "DevOps Engineer specializing in AWS, Terraform, CI/CD pipelines, Docker, Kubernetes, and Linux server administration.",
            "Data Scientist proficient in Python, R, Scikit-Learn, Pandas, NLP, and Deep Learning. Strong mathematical and statistical modeling background.",
            "Business Analyst skilled in Excel, SQL, Agile methodology, JIRA, and requirements gathering. Adept at stakeholder management.",
            "Full Stack Developer proficient in React, Node.js, Express, MongoDB, JavaScript, and AWS deployment.",
            "Cloud Architect with expertise in Azure, GCP, Cloud Security, Infrastructure as Code, and enterprise network design.",
            "Machine Learning Engineer experienced in PyTorch, MLOps, Model Deployment, Fast API, Docker, and MLflow."
        ] * 30,  # Duplicate to simulate larger sample size
        "job_role": [
            "Data Scientist", "Data Analyst", "Software Engineer", "Frontend Developer",
            "DevOps Engineer", "Data Scientist", "Business Analyst", "Software Engineer",
            "DevOps Engineer", "Data Scientist"
        ] * 30
    }
    df = pd.DataFrame(sample_data)
    df.to_csv(data_path, index=False)
else:
    df = pd.read_csv(data_path)

print(f"Dataset Loaded: {len(df)} records across {df['job_role'].nunique()} categories.")

# 3. Preprocessing & Label Encoding
X = df["resume_text"]
y = df["job_role"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Pipeline Construction
pipeline = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english"),
    LogisticRegression(C=1.0, max_iter=1000)
)

# 5. Training & Evaluation
print("Training TF-IDF + Logistic Regression Model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Save Artifacts
joblib.dump(pipeline, "models/pipeline.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("Model pipeline and label encoder saved to models/")
