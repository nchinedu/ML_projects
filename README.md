# Wine and Breast Cancer Analysis Web App

This project implements machine learning models for classification (wine and breast cancer datasets), clustering (breast cancer dataset), and a web-based breast cancer prediction system using a Random Forest Classifier. The prediction system is integrated into the home page, using the top 5 features based on feature importance. A concise PDF report summarizes classification and prediction observations.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wine_breast_cancer_ml
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train and save the Random Forest model:
   ```bash
   python train_random_forest_model.py
   ```

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open `http://localhost:5000` to view analysis, use the prediction form, or toggle/download the PDF report.

## Project Structure

- `logistic_regression_wine.py`: Logistic Regression on wine dataset.
- `svm_wine.py`: SVM on wine dataset.
- `decision_tree_wine.py`: Decision Tree on wine dataset.
- `breast_cancer_experiment.py`: Classification on breast cancer dataset.
- `kmeans_breast_cancer.py`: K-Means on breast cancer dataset.
- `agglomerative_breast_cancer.py`: Agglomerative Clustering on breast cancer dataset.
- `dbscan_breast_cancer.py`: DBSCAN on breast cancer dataset.
- `gmm_breast_cancer.py`: GMM on breast cancer dataset.
- `train_random_forest_model.py`: Train and save Random Forest model with feature importance.
- `app.py`: Flask web app with prediction in home route.
- `templates/index.html`: Main page with prediction form.
- `static/`
  - `script.js`: JavaScript for form validation.
  - `report.pdf`: PDF report for observations.
- `models/`
  - `random_forest_model.joblib`: Saved Random Forest model.
  - `scaler.joblib`: Saved scaler.
  - `top_feature_indices.joblib`: Indices of top 5 features.
  - `top_feature_names.joblib`: Names of top 5 features.
- `report.tex`: LaTeX source for PDF report.
- `requirements.txt`: Dependencies.
- `README.md`: Documentation.

## Deployment to GitHub

1. Initialize a Git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Update prediction to use top 5 features"
   ```

2. Push to GitHub:
   ```bash
   git remote add origin <repository-url>
   git push -u origin main
   ```

3. Deploy to Render:
   - Connect your GitHub repository.
   - Build command: `pip install -r requirements.txt && python train_random_forest_model.py`
   - Start command: `gunicorn app:app`
   - Access the public URL (e.g., `https://your-app.onrender.com`).

## Results

### Wine Dataset (Classification)
- Logistic Regression: ~0.9815
- SVM: ~0.7593
- Decision Tree (random_state=42): ~0.9630
- Decision Tree (random_state=None): ~0.9444 (varies)

### Breast Cancer Dataset (Classification)
- Logistic Regression: ~0.9474
- SVM: ~0.9123
- Decision Tree (random_state=42): ~0.9181
- Decision Tree (random_state=None): ~0.9064 (varies)
- Random Forest: ~0.9591

### Breast Cancer Dataset (Clustering)
- K-Means (random_state=42): ~0.3512
- K-Means (random_state=None): ~0.3512 (varies)
- Agglomerative Clustering: ~0.3468
- DBSCAN (eps=3.0, min_samples=5): ~-0.2000
- Gaussian Mixture Model: ~0.3495

## Prediction System
- **Model**: Random Forest Classifier (~0.9591 accuracy).
- **Functionality**: Integrated into home page; users input top 5 features for prediction (Malignant/Benign) with confidence.
- **Features**: Feature importance reduces inputs to 5, responsive design, client-side validation.

## Observations
- **Report**: Concise PDF (`static/report.pdf`) summarizes classification and prediction findings.
- **Classification**: Random Forest excels for breast cancer (~0.9591); Logistic Regression for wine (~0.9815).
- **Prediction**: Simplified to 5 inputs using top features; fast, reliable predictions.
- **Web App**: Prediction form in home page, toggleable PDF report, download option.