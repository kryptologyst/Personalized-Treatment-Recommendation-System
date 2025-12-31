# Personalized Treatment Recommendation System

A research-ready implementation of personalized treatment recommendation using advanced NLP techniques for healthcare applications.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research demonstration system only.**

- **NOT intended for clinical use or medical diagnosis**
- **NOT a substitute for professional medical advice**
- **All recommendations should be reviewed by qualified healthcare professionals**
- **Results are for educational and research purposes only**

## Overview

This system demonstrates state-of-the-art natural language processing techniques for personalized treatment recommendation in healthcare. It includes multiple models, comprehensive evaluation metrics, safety features, and an interactive demo interface.

### Key Features

- **Multiple Models**: TF-IDF baseline, ClinicalBERT, BioGPT, and Dual Encoder models
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, AUC, calibration metrics
- **Safety Features**: De-identification, input validation, safety checks
- **Explainability**: Feature importance and attention visualization
- **Interactive Demo**: User-friendly Streamlit interface
- **Production Ready**: Proper project structure, configuration management, logging

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data processing utilities
│   ├── metrics/           # Evaluation metrics
│   ├── utils/             # Utility functions
│   ├── train/             # Training scripts
│   └── eval/              # Evaluation scripts
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── assets/               # Generated assets (plots, reports)
├── logs/                 # Log files
├── checkpoints/          # Model checkpoints
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Personalized-Treatment-Recommendation-System.git
   cd Personalized-Treatment-Recommendation-System
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python 0470.py
   ```

## Quick Start

### 1. Run the Basic Demo

```bash
python 0470.py
```

This will:
- Generate synthetic clinical data
- Test multiple models (TF-IDF, ClinicalBERT, Dual Encoder)
- Display treatment recommendations
- Show evaluation metrics

### 2. Launch Interactive Demo

```bash
streamlit run demo/app.py
```

This opens a web interface where you can:
- Enter patient profiles
- Select different models
- View treatment recommendations
- See confidence scores and explanations
- Compare model performance

### 3. Train Models

```bash
python scripts/train.py --config configs/config.yaml
```

## Models

### 1. TF-IDF Baseline
- Traditional text similarity using TF-IDF vectors
- Fast and interpretable
- Good baseline for comparison

### 2. ClinicalBERT
- Pre-trained clinical language model
- Fine-tuned for medical text understanding
- High performance on clinical tasks

### 3. BioGPT
- Biomedical language model
- Generative capabilities
- Good for treatment description understanding

### 4. Dual Encoder
- Separate encoders for patient profiles and treatments
- Efficient similarity computation
- Scalable to large treatment databases

## Configuration

The system uses YAML configuration files. Key settings in `configs/config.yaml`:

```yaml
model:
  name: "clinicalbert-base"
  max_length: 512
  batch_size: 16
  learning_rate: 2e-5

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  max_samples: 1000

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1", "auc", "calibration"]
  top_k_recommendations: [1, 3, 5]
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Recommendation Metrics
- **Hit Rate @ K**: Percentage of patients with correct treatment in top-K recommendations
- **NDCG**: Normalized Discounted Cumulative Gain
- **Diversity**: Intra-list and inter-list diversity measures

### Clinical Metrics
- **Condition-Treatment Alignment**: How well recommendations match patient conditions
- **Confidence Calibration**: Reliability of confidence scores
- **Treatment Coverage**: Diversity of recommended treatments

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Average calibration error
- **Maximum Calibration Error (MCE)**: Maximum calibration error

## Safety Features

### De-identification
- Automatic detection and removal of PHI patterns
- Phone numbers, SSNs, dates, emails
- Configurable de-identification rules

### Input Validation
- Text length limits
- Content safety checks
- Malicious input detection

### Compliance
- Clear disclaimers in all interfaces
- No PHI logging
- Research-only usage restrictions

## Usage Examples

### Basic Usage

```python
from src.models import TreatmentRecommenderFactory
from src.data import TreatmentDataProcessor

# Load configuration
config = load_config("configs/config.yaml")

# Create model
model = TreatmentRecommenderFactory.create_model("clinicalbert", config.model)

# Get recommendations
patient_text = "45-year-old male with type 2 diabetes..."
treatments = [{"name": "Metformin", "description": "..."}]

recommendations = model.recommend_treatments(patient_text, treatments, top_k=5)
```

### Advanced Usage

```python
from src.metrics import ModelEvaluator
from src.utils import SafetyChecker

# Initialize evaluator
evaluator = ModelEvaluator(config.evaluation)

# Evaluate model
metrics = evaluator.evaluate_model(model, test_data, treatment_database)

# Safety checks
safety_checker = SafetyChecker(config)
is_safe = safety_checker.check_input_safety(patient_text)
```

## API Reference

### Models

#### `TreatmentRecommenderFactory`
- `create_model(model_type, config)`: Create a model instance
- `get_available_models()`: Get list of available model types

#### `BaseTreatmentRecommender`
- `forward(patient_text, treatment_text)`: Compute similarity score
- `recommend_treatments(patient_text, treatments, top_k)`: Get top-K recommendations

### Data Processing

#### `TreatmentDataProcessor`
- `generate_synthetic_data(num_samples)`: Generate synthetic clinical data
- `preprocess_text(text, max_length)`: Preprocess text for model input
- `split_data(data)`: Split data into train/val/test sets

#### `DeidentificationProcessor`
- `deidentify_text(text)`: Remove PHI from text
- `check_phi_content(text)`: Check if text contains PHI

### Evaluation

#### `ModelEvaluator`
- `evaluate_model(model, test_data, treatments)`: Comprehensive model evaluation
- `create_evaluation_report(metrics, model_name)`: Generate evaluation report

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{personalized_treatment_recommendation,
  title={Personalized Treatment Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Personalized-Treatment-Recommendation-System}
}
```

## Acknowledgments

- ClinicalBERT: [Emily Alsentzer et al.](https://arxiv.org/abs/1904.05342)
- BioGPT: [Microsoft Research](https://arxiv.org/abs/2210.10341)
- Transformers library: [Hugging Face](https://huggingface.co/transformers/)

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

## Changelog

### Version 1.0.0
- Initial release
- Multiple model implementations
- Comprehensive evaluation metrics
- Interactive demo interface
- Safety and compliance features

---

**Remember: This system is for research and educational purposes only. It is not intended for clinical use or medical diagnosis.**
# Personalized-Treatment-Recommendation-System
