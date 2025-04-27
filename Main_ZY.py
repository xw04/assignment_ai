import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from joblib import load

# --- Functions ---

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    return accuracy, precision, recall, f1, roc_auc, y_pred, y_prob

def get_model(model_option):
    if model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        rf = load('random_forest_model.joblib')
        prediction_rf = rf.predict(X_test)
    elif model_option == "Gradient Boosting Classifier":
        model = SVC(probability=True, random_state=42)
        gbc = load('gradient_boosting_model.joblib')
        prediction_gbc = gbc.predict(X_test)
    elif model_option == "Naive Bayes":
        model = GaussianNB(priors=[0.5, 0.5])
        gaussian_nb_loaded = load('gaussian_nb_model.joblib')
        prediction_nb = gaussian_nb_loaded.predict(X_test)
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_classifier_loaded = load('xgb_classifier_model.joblib')
        prediction_xgb = xgb_classifier_loaded.predict(X_test)
    return model

def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdens_j = tpr - fpr
    best_threshold = thresholds[np.argmax(youdens_j)]
    return best_threshold

# --- Main App ---

# Title
st.title("🏦 Credit Risk Prediction Dashboard")

# Sidebar - Settings
st.sidebar.header("🔍 Model and Input Settings")
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "SVM", "Naive Bayes", "XGBoost"])
apply_pca = st.sidebar.checkbox("Apply PCA", value=True)
pca_mode = st.sidebar.radio("PCA Mode", ["Manual", "Auto (95% Variance)"])
if pca_mode == "Manual":
    n_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=10, value=5)

# Load Data
df = load_data()
df = df.assign(
    person_emp_length=df['person_emp_length'].fillna(df['person_emp_length'].median()),
    loan_int_rate=df['loan_int_rate'].fillna(df['loan_int_rate'].median())
)

# Data Split
X = df.drop(columns=['loan_status'], axis=1)
X = X.select_dtypes(include=[np.number])
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA if selected
if apply_pca:
    if pca_mode == "Manual":
        pca = PCA(n_components=n_components)
    else:  # Auto (keep 95% variance)
        pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

# Get Model
model = get_model(model_option)
model.fit(X_train, y_train)

# Predict Probabilities
accuracy_default, precision_default, recall_default, f1_default, roc_auc_default, y_test_pred_default, y_prob = evaluate_model(model, X_test, y_test, 0.5)

# Find Optimal Threshold (Youden's J)
optimal_threshold = find_optimal_threshold(y_test, y_prob)

# Sidebar - Threshold Tuning
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, optimal_threshold, 0.01)
st.sidebar.markdown(f"🧠 **Recommended Optimal Threshold (Youden's J): {optimal_threshold:.2f}**")

# Evaluate with selected threshold
accuracy, precision, recall, f1, roc_auc, y_test_pred, _ = evaluate_model(model, X_test, y_test, threshold)

# Input Form
st.sidebar.header("📝 Input Features")
with st.sidebar.form(key="input_form"):
    person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=18, step=1)
    person_income = st.sidebar.number_input("Income ($)", min_value=0.0, value=0.0)
    person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=5, step=1)
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0.0, value=0.0)
    loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=0.0)
    if person_income > 0:
        loan_percent_income = (loan_amnt / person_income) * 100
    else:
        loan_percent_income = 0.0
    
    st.sidebar.number_input(
        "Loan Percent Income (%)",
        value=loan_percent_income,
        format="%.2f",
        disabled=True
    )
    cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=10, step=1)
    submit_button = st.form_submit_button(label="Predict")

# Prepare Input
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_emp_length': [person_emp_length],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length],
})

input_data_scaled = scaler.transform(input_data)
if apply_pca:
    input_data_scaled = pca.transform(input_data_scaled)

# Prediction
if submit_button:
    probability = model.predict_proba(input_data_scaled)
    prediction = (probability[:,1] >= threshold).astype(int)

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 0:
        st.success("✅ **Low Risk**")
    else:
        st.error("⚠️ **High Risk**")

    st.write(f"Low Risk Probability: **{probability[0][0]*100:.2f}%**")
    st.write(f"High Risk Probability: **{probability[0][1]*100:.2f}%**")
    st.write(f"Applied Threshold: **{threshold:.2f}**")

# Show Metrics
st.subheader(f"📊 {model_option} Model Performance (Threshold = {threshold:.2f})")
st.table(pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
    'Score': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{roc_auc:.4f}"]
}))

# Confusion Matrix
st.subheader("🧩 Confusion Matrix on Test Set")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low Risk", "High Risk"], yticklabels=["Low Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc_default:.2f})")
ax2.plot([0, 1], [0, 1], color='grey', linestyle='--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic (ROC)")
ax2.legend()
st.pyplot(fig2)
