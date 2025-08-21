import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, precision_score,
                             recall_score, confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
import joblib
import shap
from fpdf import FPDF

# 1. Load Data
df = pd.read_csv(r"C:\Users\samee\Desktop\projects\ml csv files\heart.csv")


# 2. Data Preprocessing
df = df.fillna(df.median(numeric_only=True))

# Detect categorical features automatically
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Encode categorical features if any
if categorical_features:
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 3. Identify Target Column
# Common names in heart datasets: target, HeartDisease, Disease, Outcome
possible_targets = ['target', 'HeartDisease', 'Disease', 'Outcome']
target_col = None
for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(f"Could not find target column. Available columns: {df.columns.tolist()}")

print(f"Using target column: {target_col}")

# 4. Feature Selection
X = df.drop(target_col, axis=1)
y = df[target_col]
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)
selector = SelectFromModel(rf_selector, threshold='median', prefit=True)
X = selector.transform(X)
selected_features = np.array(df.drop(target_col, axis=1).columns)[selector.get_support()]


# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Handle Imbalance (SMOTE)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Original training set shape:", X_train.shape)
print("Resampled training set shape:", X_train_res.shape)
# Scale train and test sets
scaler = StandardScaler()

# Fit on training (resampled) data
X_train_res_scaled = scaler.fit_transform(X_train_res)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# 7. Model Definitions & Hyperparameter Tuning (RandomizedSearchCV Example for RF)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_cv = RandomizedSearchCV(rf, rf_params, cv=StratifiedKFold(n_splits=5), n_iter=10, scoring='roc_auc', n_jobs=-1, verbose=2, random_state=42)
rf_cv.fit(X_train_res_scaled, y_train_res)
best_rf = rf_cv.best_estimator_
print("Best RF Parameters:", rf_cv.best_params_)

# 8. Model Calibration
calibrated_rf = CalibratedClassifierCV(best_rf, method='sigmoid', cv=5)
calibrated_rf.fit(X_train_res_scaled, y_train_res)

# 9. Evaluation on Test Set
y_pred = calibrated_rf.predict(X_test_scaled)
y_proba = calibrated_rf.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(calibrated_rf, X_train_res_scaled, y_train_res, cv=cv, scoring='roc_auc')
print("Stratified CV ROC-AUC Mean:", cv_scores.mean())

# 11. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png")
plt.show()

# 12. Interactive ROC Curve (Plotly)
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(title='Interactive ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
fig_roc.write_html("roc_curve_interactive.html")
fig_roc.show()

# 13. Interactive Feature Importance (Plotly)
feat_importances = best_rf.feature_importances_
fig_feat = px.bar(x=selected_features, y=feat_importances, labels={'x':'Feature', 'y':'Importance'}, title='Feature Importances (Interactive)')
fig_feat.write_html("feature_importance_interactive.html")
fig_feat.show()

# 14. SHAP Summary Plot
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_scaled)
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_test_scaled, feature_names=selected_features, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.show()

# 15. Save Model, Scaler, Features
joblib.dump(calibrated_rf, "heart_disease_rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
np.save("selected_features.npy", selected_features)

# 16. Custom PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Heart Disease Prediction Report", ln=True, align='C')
pdf.cell(200, 10, txt=f"Accuracy: {accuracy_score(y_test, y_pred):.2f}", ln=True)
pdf.cell(200, 10, txt=f"Precision: {precision_score(y_test, y_pred):.2f}", ln=True)
pdf.cell(200, 10, txt=f"Recall: {recall_score(y_test, y_pred):.2f}", ln=True)
pdf.cell(200, 10, txt=f"ROC-AUC: {roc_auc_score(y_test, y_proba):.2f}", ln=True)
pdf.cell(200, 10, txt="See saved plots for visualizations.", ln=True)
pdf.output("custom_report.pdf")

# 17. Predict on New Data Example
def predict_new_patient(patient_data_dict):
    new_data_df = pd.DataFrame([patient_data_dict])
    new_data_selected = new_data_df[selected_features]
    new_data_scaled = scaler.transform(new_data_selected)
    prediction = calibrated_rf.predict(new_data_scaled)
    prob = calibrated_rf.predict_proba(new_data_scaled)[0][1]
    return prediction[0], prob

# Example usage:
# patient_data_dict = {feat: value, ...}
# pred, prob = predict_new_patient(patient_data_dict)
# print(f"Prediction: {pred} (Prob: {prob:.2f})")

print("\nAll outputs (plots, HTMLs, report, models) are saved in your workspace. Open HTML files for interactive plots, PNGs for static plots, and PDF for the report.")