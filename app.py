import streamlit as st
import joblib
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="Transparent AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/pipeline.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return pipeline, label_encoder

try:
    pipeline, label_encoder = load_artifacts()
    class_names = list(label_encoder.classes_)
except Exception as e:
    st.error("Model artifacts not found! Please run `python train.py` first.")
    st.stop()

# Header Section
st.title("📄 Transparent AI for Recruitment & Resume Screening")
st.markdown("""
This system predicts the target job role for a given resume using Machine Learning **and** provides 
transparent explanations using **LIME (Locally Interpretable Model-Agnostic Explanations)** to explain *why* specific words influenced the prediction.
""")

st.divider()

# Input UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Candidate Resume")
    default_text = """Senior Data Scientist with 5 years experience in Python, SQL, and Machine Learning. 
Proficient in TensorFlow, Pandas, Scikit-Learn, and building NLP pipelines. Strong background in statistical modeling."""
    
    resume_input = st.text_area("Paste Resume Text Here:", value=default_text, height=280)
    num_features = st.slider("Number of Explanation Keywords to Display:", min_value=5, max_value=20, value=10)
    analyze_btn = st.button("Analyze & Explain Resume", type="primary")

# Analysis & Explanation
if analyze_btn and resume_input.strip():
    with col2:
        st.subheader("2. Prediction & Confidence")
        
        # Predict Class Probabilities
        probs = pipeline.predict_proba([resume_input])[0]
        pred_class_idx = probs.argmax()
        pred_label = class_names[pred_class_idx]
        confidence = probs[pred_class_idx] * 100
        
        st.success(f"**Predicted Category:** {pred_label}")
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        
        # Display Top 3 Probabilities
        top_3_indices = probs.argsort()[-3:][::-1]
        st.markdown("**Top Category Probabilities:**")
        for idx in top_3_indices:
            st.progress(float(probs[idx]), text=f"{class_names[idx]}: {probs[idx]*100:.1f}%")

    st.divider()
    st.subheader("3. Explainable AI (XAI) Word Attribution")
    st.write(f"The chart below highlights key terms that contributed **positively** (supported `{pred_label}`) or **negatively** against the prediction:")

    with st.spinner("Generating LIME Explanation..."):
        # Initialize LIME Text Explainer
        explainer = LimeTextExplainer(class_names=class_names)
        
        # Define prediction function for LIME
        def predict_proba_fn(texts):
            return pipeline.predict_proba(texts)
        
        exp = explainer.explain_instance(
            resume_input,
            predict_proba_fn,
            num_features=num_features,
            labels=[pred_class_idx]
        )

        # Render LIME Explanation Plot
        fig = exp.as_pyplot_figure(label=pred_class_idx)
        st.pyplot(fig)

        # Render Interactive HTML Heatmap
        st.subheader("Interactive Feature Highlight:")
        html_content = exp.as_html(labels=[pred_class_idx])
        components.html(html_content, height=350, scrolling=True)
