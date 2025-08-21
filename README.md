# Heart Disease Prediction Pipeline

## Overview

This repository provides a comprehensive machine learning pipeline for heart disease risk prediction using clinical data. The pipeline includes data preprocessing, feature selection, class balancing, model training (Random Forest, Logistic Regression, SVM), hyperparameter tuning, calibration, evaluation, interpretability, and deployment. Results are visualized and summarized in custom reports.

---

## Dataset

- **File:** `heart.csv`  
- **Source:** [Kaggle - Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- **Columns:**  
  `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease`
- **How to use:**  
  Download `heart.csv` from Kaggle and place it in your project directory.

---

## Features

- **Data Preprocessing:** Handles missing values, encodes categorical features.
- **Feature Selection:** Identifies important features using Random Forest.
- **Class Balancing:** SMOTE for minority class oversampling.
- **Scaling:** StandardScaler for feature normalization.
- **Model Training:** Random Forest, Logistic Regression, SVM.
- **Hyperparameter Tuning:** RandomizedSearchCV for optimal parameters.
- **Calibration:** CalibratedClassifierCV for reliable probability outputs.
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Cross-validation.
- **Interpretability:**  
  - Feature Importance (static and interactive)
  - SHAP plots for global and local explanation
- **Visualizations:**  
  - Confusion matrix heatmap
  - ROC curve (matplotlib & interactive Plotly)
  - Feature importance bar chart
- **Reporting:**  
  - PDF summary report
- **Deployment:**  
  - Model, scaler, and selected features saved for future use.
  - Prediction function for new patient data.

---

## Quick Start

### 1. Install Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn joblib shap fpdf
```

### 2. Prepare Dataset

- Download `heart.csv` from [Kaggle - Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci).
- Place `heart.csv` in your project directory.

### 3. Run the Pipeline

```bash
python heart_disease_full_visual_pipeline.py
```

### 4. Outputs

- **Visualizations:**  
  - `confusion_matrix_heatmap.png`
  - `roc_curve_interactive.html`
  - `feature_importance_interactive.html`
  - `shap_summary_plot.png`
- **Report:**  
  - `custom_report.pdf`
- **Model Files:**  
  - `heart_disease_rf_model.joblib`
  - `scaler.joblib`
  - `selected_features.npy`

### 5. Predict New Data

Use the `predict_new_patient` function in the script to estimate heart disease risk for new clinical records.

---

## File Structure

```
.
├── heart_disease_full_visual_pipeline.py   # Main script
├── heart.csv                              # Dataset
├── confusion_matrix_heatmap.png           # Visualization
├── roc_curve_interactive.html             # Interactive ROC
├── feature_importance_interactive.html    # Interactive Feature Plot
├── shap_summary_plot.png                  # SHAP explanation
├── custom_report.pdf                      # PDF summary
├── heart_disease_rf_model.joblib          # Trained model
├── scaler.joblib                          # Scaler
├── selected_features.npy                  # Features used
├── README.md                              # This file
├── LICENSE                                # MIT License
```

---

## References

- [Kaggle: Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.
