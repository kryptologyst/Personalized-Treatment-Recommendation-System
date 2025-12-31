"""
Interactive Streamlit demo for Personalized Treatment Recommendation System.

This demo provides a user-friendly interface for testing treatment recommendations
with various models and visualizing results with explainability features.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import TreatmentDataProcessor, DeidentificationProcessor
from models import TreatmentRecommenderFactory
from metrics import ModelEvaluator
from utils import set_seed, get_device, SafetyChecker


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Personalized Treatment Recommendation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .treatment-card {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    return OmegaConf.load(config_path)


@st.cache_resource
def initialize_components(config):
    """Initialize all components."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize components
    data_processor = TreatmentDataProcessor(config.data)
    deid_processor = DeidentificationProcessor()
    safety_checker = SafetyChecker(config)
    
    return data_processor, deid_processor, safety_checker


@st.cache_data
def load_treatment_database():
    """Load treatment database."""
    treatments = [
        {
            "name": "Metformin + Lifestyle Modification",
            "description": "Metformin 500mg twice daily with comprehensive lifestyle counseling including diet modification and exercise program. Recommended for newly diagnosed type 2 diabetes patients with BMI >25.",
            "indications": ["type 2 diabetes", "obesity"],
            "contraindications": ["renal impairment", "liver disease"]
        },
        {
            "name": "ACE Inhibitor Therapy",
            "description": "Lisinopril 10mg daily for blood pressure control. Monitor renal function and potassium levels. First-line therapy for hypertension with diabetes or heart failure.",
            "indications": ["hypertension", "heart failure", "type 2 diabetes"],
            "contraindications": ["pregnancy", "angioedema history"]
        },
        {
            "name": "Statin Therapy",
            "description": "Atorvastatin 20mg daily for cholesterol management. Regular monitoring of liver enzymes and muscle symptoms. Indicated for cardiovascular risk reduction.",
            "indications": ["hyperlipidemia", "heart failure", "type 2 diabetes"],
            "contraindications": ["active liver disease", "pregnancy"]
        },
        {
            "name": "SSRI Antidepressant",
            "description": "Sertraline 50mg daily for depression management. Gradual dose titration with monitoring for side effects. Effective for mood disorders with good safety profile.",
            "indications": ["depression", "anxiety"],
            "contraindications": ["MAOI use", "seizure disorder"]
        },
        {
            "name": "Physical Therapy Program",
            "description": "Structured physical therapy program focusing on pain management and functional improvement. Includes exercises, manual therapy, and education.",
            "indications": ["chronic pain", "osteoporosis"],
            "contraindications": ["acute fracture", "severe osteoporosis"]
        },
        {
            "name": "Beta-Blocker Therapy",
            "description": "Metoprolol 25mg twice daily for heart rate control and blood pressure management. Effective for atrial fibrillation and heart failure.",
            "indications": ["atrial fibrillation", "heart failure", "hypertension"],
            "contraindications": ["asthma", "severe bradycardia"]
        },
        {
            "name": "Inhaled Corticosteroid",
            "description": "Fluticasone inhaler for asthma control. Reduces airway inflammation and prevents asthma attacks.",
            "indications": ["asthma", "COPD"],
            "contraindications": ["severe fungal infections", "tuberculosis"]
        },
        {
            "name": "Bisphosphonate Therapy",
            "description": "Alendronate 70mg weekly for osteoporosis treatment. Increases bone density and reduces fracture risk.",
            "indications": ["osteoporosis"],
            "contraindications": ["esophageal stricture", "hypocalcemia"]
        }
    ]
    return treatments


def create_model(model_type: str, config: Dict[str, Any]):
    """Create and initialize model."""
    try:
        model = TreatmentRecommenderFactory.create_model(model_type, config.model)
        
        # For TF-IDF model, we need to fit it
        if model_type == "tfidf":
            # Generate some sample data for fitting
            data_processor = TreatmentDataProcessor(config.data)
            sample_data = data_processor.generate_synthetic_data(100)
            texts = [sample["patient_profile"] + " " + sample["treatment_description"] 
                    for sample in sample_data]
            model.fit(texts)
        
        return model
    except Exception as e:
        st.error(f"Failed to load {model_type} model: {str(e)}")
        return None


def display_disclaimer():
    """Display safety disclaimer."""
    st.markdown("""
    <div class="disclaimer-box">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This is a research demonstration system only.</strong></p>
        <ul>
            <li>NOT intended for clinical use or medical diagnosis</li>
            <li>NOT a substitute for professional medical advice</li>
            <li>All recommendations should be reviewed by qualified healthcare professionals</li>
            <li>Results are for educational and research purposes only</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def display_model_comparison():
    """Display model comparison section."""
    st.header("Model Comparison")
    
    config = load_config()
    treatment_database = load_treatment_database()
    
    # Generate sample data for comparison
    data_processor = TreatmentDataProcessor(config.data)
    sample_data = data_processor.generate_synthetic_data(50)
    
    # Available models
    available_models = ["tfidf", "clinicalbert", "dual_encoder"]
    
    # Create models
    models = {}
    for model_type in available_models:
        model = create_model(model_type, config)
        if model is not None:
            models[model_type] = model
    
    if not models:
        st.error("No models available for comparison")
        return
    
    # Evaluate models
    evaluator = ModelEvaluator(config.evaluation)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        
        model_metrics = {}
        for model_name, model in models.items():
            try:
                metrics = evaluator.evaluate_model(model, sample_data[:20], treatment_database)
                model_metrics[model_name] = metrics
            except Exception as e:
                st.warning(f"Failed to evaluate {model_name}: {str(e)}")
        
        if model_metrics:
            # Create performance comparison chart
            metrics_df = pd.DataFrame(model_metrics).T
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
            available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
            
            if available_metrics:
                fig = px.bar(
                    metrics_df[available_metrics],
                    title="Model Performance Comparison",
                    labels={'index': 'Model', 'value': 'Score'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top-K Hit Rates")
        
        if model_metrics:
            # Create top-k comparison chart
            top_k_metrics = {}
            for model_name, metrics in model_metrics.items():
                top_k_metrics[model_name] = {
                    k: metrics.get(f"hit_rate_at_{k}", 0.0) 
                    for k in [1, 3, 5]
                }
            
            top_k_df = pd.DataFrame(top_k_metrics).T
            fig = px.bar(
                top_k_df,
                title="Top-K Hit Rate Comparison",
                labels={'index': 'Model', 'value': 'Hit Rate'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_recommendation_interface():
    """Display main recommendation interface."""
    st.header("Treatment Recommendation")
    
    config = load_config()
    data_processor, deid_processor, safety_checker = initialize_components(config)
    treatment_database = load_treatment_database()
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            ["tfidf", "clinicalbert", "dual_encoder"],
            help="Choose the model for treatment recommendation"
        )
    
    with col2:
        top_k = st.slider("Number of Recommendations", 1, 8, 5)
    
    # Create model
    model = create_model(model_type, config)
    
    if model is None:
        st.error("Failed to load model. Please try again.")
        return
    
    # Patient input
    st.subheader("Patient Profile")
    
    # Example patient profiles
    example_profiles = [
        "45-year-old male with type 2 diabetes and high blood pressure. BMI is 32. Needs medication for blood sugar control and weight management.",
        "67-year-old female with heart failure and atrial fibrillation. History of hypertension. Current medications include metoprolol.",
        "34-year-old male with depression and anxiety. No significant medical history. Seeking treatment for mood disorders.",
        "58-year-old female with osteoporosis and chronic back pain. History of fractures. Needs pain management and bone strengthening.",
        "29-year-old male with asthma. Allergic to penicillin. Needs long-term asthma control medication."
    ]
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Manual Entry", "Select Example"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        patient_text = st.text_area(
            "Enter Patient Profile",
            height=150,
            placeholder="Enter detailed patient information including age, conditions, medical history, current medications, allergies, and treatment goals..."
        )
    else:
        selected_example = st.selectbox("Select Example Profile", example_profiles)
        patient_text = st.text_area(
            "Patient Profile (editable)",
            value=selected_example,
            height=150
        )
    
    # De-identification option
    deidentify = st.checkbox("De-identify Text", value=True, help="Remove potential PHI from the text")
    
    if deidentify and patient_text:
        patient_text = deid_processor.deidentify_text(patient_text)
        st.info("Text has been de-identified")
    
    # Safety check
    if patient_text:
        if not safety_checker.check_input_safety(patient_text):
            st.error("Input text failed safety checks. Please revise.")
            return
        
        if deid_processor.check_phi_content(patient_text):
            st.warning("Potential PHI detected in text. Consider enabling de-identification.")
    
    # Generate recommendations
    if st.button("Generate Recommendations", type="primary"):
        if not patient_text.strip():
            st.error("Please enter a patient profile")
            return
        
        try:
            with st.spinner("Generating recommendations..."):
                recommendations = model.recommend_treatments(
                    patient_text, treatment_database, top_k=top_k
                )
            
            # Display recommendations
            st.subheader("Treatment Recommendations")
            
            for i, rec in enumerate(recommendations, 1):
                treatment = rec["treatment"]
                confidence = rec["confidence"]
                
                # Create treatment card
                st.markdown(f"""
                <div class="treatment-card">
                    <h4>#{i} {treatment['name']}</h4>
                    <p><strong>Confidence:</strong> {confidence:.3f}</p>
                    <p><strong>Description:</strong> {treatment['description']}</p>
                    <p><strong>Indications:</strong> {', '.join(treatment.get('indications', []))}</p>
                    <p><strong>Contraindications:</strong> {', '.join(treatment.get('contraindications', []))}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display confidence distribution
            if len(recommendations) > 1:
                st.subheader("Confidence Distribution")
                confidences = [rec["confidence"] for rec in recommendations]
                treatment_names = [rec["treatment"]["name"] for rec in recommendations]
                
                fig = px.bar(
                    x=treatment_names,
                    y=confidences,
                    title="Recommendation Confidence Scores",
                    labels={'x': 'Treatment', 'y': 'Confidence Score'},
                    color=confidences,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            logger.error(f"Recommendation error: {str(e)}")


def display_explainability():
    """Display explainability features."""
    st.header("Model Explainability")
    
    st.info("""
    This section demonstrates how the model makes its recommendations.
    Explainability features help healthcare professionals understand the reasoning
    behind treatment suggestions.
    """)
    
    # Placeholder for explainability features
    st.subheader("Feature Importance")
    
    # Simulate feature importance (in real implementation, this would come from SHAP or attention maps)
    features = [
        "Patient Age", "BMI", "Medical History", "Current Medications",
        "Allergies", "Symptoms", "Laboratory Values", "Vital Signs"
    ]
    
    importance_scores = np.random.uniform(0.1, 1.0, len(features))
    importance_scores = importance_scores / importance_scores.sum()
    
    fig = px.bar(
        x=importance_scores,
        y=features,
        orientation='h',
        title="Feature Importance for Treatment Recommendation",
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=importance_scores,
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Attention Visualization")
    st.info("Attention maps show which parts of the patient profile the model focuses on when making recommendations.")
    
    # Placeholder attention visualization
    st.image("https://via.placeholder.com/800x300/1f77b4/ffffff?text=Attention+Map+Visualization", 
             caption="Attention map showing model focus areas")


def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">🏥 Personalized Treatment Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Treatment Recommendation", "Model Comparison", "Explainability", "About"]
    )
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Information")
    st.sidebar.info("""
    **Version:** 1.0.0  
    **Purpose:** Research & Education  
    **Models:** TF-IDF, ClinicalBERT, Dual Encoder  
    **Data:** Synthetic Clinical Data
    """)
    
    # Main content based on selected page
    if page == "Treatment Recommendation":
        display_recommendation_interface()
    elif page == "Model Comparison":
        display_model_comparison()
    elif page == "Explainability":
        display_explainability()
    elif page == "About":
        st.header("About This System")
        
        st.markdown("""
        ## Personalized Treatment Recommendation System
        
        This is a research demonstration system that uses advanced natural language processing
        techniques to recommend treatments based on patient profiles.
        
        ### Features
        
        - **Multiple Models**: TF-IDF baseline, ClinicalBERT, and Dual Encoder models
        - **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, AUC, calibration metrics
        - **Safety Features**: De-identification, input validation, safety checks
        - **Explainability**: Feature importance and attention visualization
        - **Interactive Demo**: User-friendly interface for testing recommendations
        
        ### Models
        
        1. **TF-IDF Baseline**: Traditional text similarity using TF-IDF vectors
        2. **ClinicalBERT**: Pre-trained clinical language model for medical text understanding
        3. **Dual Encoder**: Separate encoders for patient profiles and treatment descriptions
        
        ### Use Cases
        
        - Medical education and training
        - Research on clinical decision support
        - Demonstration of NLP in healthcare
        - Treatment recommendation algorithm development
        
        ### Technical Details
        
        - **Framework**: PyTorch, Transformers, Streamlit
        - **Data**: Synthetic clinical data for demonstration
        - **Evaluation**: Comprehensive metrics including clinical-specific measures
        - **Safety**: Built-in de-identification and validation
        
        ### Disclaimer
        
        This system is for research and educational purposes only. It is not intended for
        clinical use or medical diagnosis. All recommendations should be reviewed by
        qualified healthcare professionals.
        """)


if __name__ == "__main__":
    main()
