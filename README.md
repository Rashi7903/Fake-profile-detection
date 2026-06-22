🛡️ DeepShield: Advanced Fake Profile Detection

An enterprise-grade Machine Learning system designed to identify fraudulent social media accounts using behavioral metadata and ensemble learning.

 🚀 Key Features (Advanced Upgrade):
- **Stacked Ensemble Pipeline**: Combines multiple feature extraction techniques with an optimized RandomForest model.
- **Automated ColumnTransformer**: Robust handling of both categorical and numerical data types using professional `scikit-learn` pipelines.
- **Dynamic Feature Engineering**: Real-time computation of follower/following ratios and name-to-username similarity metrics.
- **GridSearchCV Optimization**: Automatically tuned hyperparameters with 5-fold Stratified Cross-Validation for maximum F1-score.
- **Interactive Dashboard**: A modern Streamlit-based UI with real-time prediction gauges and detailed performance analytics (Plotly).
- **Comprehensive Evaluation**: Automated generation of ROC curves, Confusion Matrices, and Feature Importance reports.

 🛠️ Technology Stack :
- **Languages**: Python 3.x
- **Core ML**: `scikit-learn`, `pandas`, `numpy`
- **Visualization**: `plotly`, `seaborn`, `matplotlib`
- **Dashboard**: `streamlit`
- **Artifacts**: `joblib` for model persistence

📂 Project Structure :
```text
├── data/               # Raw training and test datasets
├── model/              # Trained pipeline (advanced_model.joblib)
├── plots/              # Generated performance visualizations
├── trainer.py          # Class-based training pipeline with GridSearchCV
├── dashboard.py        # Streamlit-powered interactive UI
├── main.py             # Optimized CLI for quick testing
└── README.md           # Documentation
```

 🚥 **How to Run :**
1. **Setup Environment**: `pip install -r requirements.txt`
2. **Train Model**: Run `python trainer.py` to optimize and save the pipeline.
3. **Launch Dashboard**: Run `streamlit run dashboard.py` for the interactive experience.
4. **CLI Mode**: Run `python main.py` for terminal-based testing.

Deployment :
https://fake-profile-detection-v8xwvlyvsrtddrrwliauxp.streamlit.app
---
