# 🏦 Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **approved** or **rejected** based on the applicant's financial and personal information. The project covers the full workflow: data cleaning, preprocessing, model training and comparison, evaluation, and deployment as an interactive **Streamlit** web app.

## 📌 Overview

Banks review a large number of loan applications, and the decision depends on many factors such as income, credit score, and assets. This project builds a classification model that learns these patterns from historical data and gives a fast, data-driven prediction together with an approval probability.

## 📊 Dataset

- **Size:** 4,269 loan applications
- **Features:** 11 input features + 1 target (`loan_status`)
- **Target:** `loan_status` (`Approved` = 1, `Rejected` = 0)

| Feature | Description |
|---|---|
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term in years |
| `cibil_score` | Credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |

The `loan_id` column is dropped because it carries no predictive information.

## ⚙️ Methodology

1. **Data cleaning:** the raw CSV has leading spaces in column names and text values (e.g. `" loan_status"`, `" Approved"`), which are stripped after loading.
2. **Preprocessing pipeline** (scikit-learn `ColumnTransformer`):
   - Numerical features: median imputation + `StandardScaler`
   - Categorical features: most-frequent imputation + `OneHotEncoder`
3. **Train/test split:** 80/20, stratified, `random_state=42`.
4. **Models compared:** Logistic Regression, Random Forest, Gradient Boosting. Each model is wrapped in a `Pipeline` with the preprocessor to avoid data leakage.
5. **Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curves, and 5-fold stratified cross-validation.
6. **Best model:** selected by F1-score and saved with `joblib`.

## 🚀 Streamlit App

The app loads the saved model and provides a form to enter an applicant's details. It returns the decision (Approved / Rejected) and the approval probability.

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📁 Project Structure

```
├── Loan Approval Prediction.ipynb   # Data analysis, training, and evaluation
├── loan_approval_dataset.csv        # Dataset
├── best_loan_model.joblib           # Saved best model
├── app.py                           # Streamlit web app
├── requirements.txt                 # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Streamlit

## 📈 Results

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9145 | 0.9210 | 0.9435 | 0.9321 | 0.9726 |
| Random Forest | 0.9813 | 0.9849 | 0.9849 | 0.9849 | 0.9986 |
| **Gradient Boosting** | **0.9824** | 0.9831 | **0.9887** | **0.9859** | 0.9976 |

**Best model:** Gradient Boosting (selected by F1-score), saved as `best_loan_model.joblib`.


## 🔍 Key Insights

- The credit score (`cibil_score`) is typically the most influential feature in the decision.
- The classes are only mildly imbalanced (about 62% approved / 38% rejected), so resampling techniques like SMOTE were not necessary.

## ⚠️ Disclaimer

This project is for educational purposes only. It should not be used as the sole basis for real-world lending decisions.

## 📄 License

This project is licensed under the MIT License.
