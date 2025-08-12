# Thyroid Cancer Recurrence Predictor

A dark-themed, AI-powered Streamlit web application to predict thyroid cancer recurrence risk using a Random Forest model trained on the Thyroid Disease dataset. The app provides a professional, user-friendly interface for clinicians and researchers to input patient data and obtain recurrence predictions with probability scores.

## Features
- **Prediction**: Input patient details (e.g., Age, Stage, Response) to predict recurrence (Yes/No) with probability.
- **Model Metrics**: Displays performance metrics (Accuracy: 97.39%, Precision: 100%, Recall: 90.63%, F1: 95.08%, ROC-AUC: 99.51%).
- **Field Explanations**: Detailed guidance on input fields (Demographics, Clinical History, Pathology/Staging).

## Dataset
The model is trained on a thyroid cancer dataset (~383 samples) with features like Age, Gender, Risk, Stage, and Response. The target variable is `Recurred` (Yes/No), with ~30% positive cases.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/thyroid-cancer-predictor.git
   cd thyroid-cancer-predictor
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Model File**:
   - Place the trained model file (`thyroid_model.pkl`) in the root directory.

5. **Directory Structure**:
   ```plaintext
   thyroid-cancer-predictor/
   ├── .streamlit/
   │   └── config.toml      # Dark theme configuration
   ├── app.py               # Streamlit app
   ├── requirements.txt     # Dependencies
   ├── thyroid_model.pkl    # Trained model
   ├── README.md            # This file
   └── dataset.csv          # Dataset (not included)
   ```

## Usage

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   Open the provided URL (e.g., `http://localhost:8501`) in your browser.

2. **Input Patient Data**:
   - Navigate to the "Predict" tab.
   - Enter patient details in the form (grouped into Demographics, Clinical History, Pathology/Staging).
   - Use tooltips and the "Field Explanations" tab for guidance.
   - Click "Predict Recurrence" to view the result (Yes/No) and probability.

3. **Interpret Results**:
   - **Prediction**: "Yes" (red) or "No" (green) indicates recurrence risk.
   - **Probability**: A progress bar shows the likelihood of recurrence (0-100%).
   - **Metrics**: Sidebar displays model performance (e.g., 97.39% accuracy).


## Model Details
- **Algorithm**: Random Forest Classifier
- **Preprocessing**:
  - Ordinal features (e.g., Risk, Stage): Encoded with `OrdinalEncoder` to preserve order.
  - Nominal features (e.g., Gender, Pathology): One-hot encoded with `OneHotEncoder`.
  - Numerical feature (Age): Scaled with `StandardScaler`.
- **Performance** (on test set, 30% split):
  - Accuracy: 97.39%
  - Precision: 100%
  - Recall: 90.63%
  - F1 Score: 95.08%
  - ROC-AUC: 99.51%

## Important Note
- **Clinical Use**: This app is for research purposes only. Always consult a medical professional for clinical decisions.
