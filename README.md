# Wine and Breast Cancer Analysis Web App

This project builds machine learning models for classification (wine and breast cancer datasets) and clustering (breast cancer dataset). Classification uses Logistic Regression, SVM, Decision Tree, and Random Forest. Clustering uses K-Means, Agglomerative Clustering, DBSCAN, and Gaussian Mixture Model (GMM). Results and observations are displayed in a Flask web app, with options to toggle classification and clustering observations.

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

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open `http://localhost:5000`. Use "Show Classification Observations" or "Show Clustering Observations" to view detailed analyses.

## Project Structure

- `logistic_regression_wine.py`: Logistic Regression on wine dataset.
- `svm_wine.py`: SVM on wine dataset.
- `decision_tree_wine.py`: Decision Tree on wine dataset.
- `breast_cancer_experiment.py`: Classification on breast cancer dataset.
- `kmeans_breast_cancer.py`: K-Means on breast cancer dataset.
- `agglomerative_breast_cancer.py`: Agglomerative Clustering on breast cancer dataset.
- `dbscan_breast_cancer.py`: DBSCAN on breast cancer dataset.
- `gmm_breast_cancer.py`: GMM on breast cancer dataset.
- `app.py`: Flask web app.
- `templates/index.html`: HTML template.
- `observations.md`: Classification observations.
- `clustering_observations.md`: Clustering observations.
- `requirements.txt`: Dependencies.
- `README.md`: Documentation.

## Deployment to GitHub

1. Initialize a Git repository:
   ```bash
   git init
   git add .
   git commit -m "Add clustering tasks and observations"
   ```

2. Push to GitHub:
   ```bash
   git remote add origin <repository-url>
   git push -u origin main
   ```

3. Deploy to Render:
   - Connect your GitHub repository.
   - Build command: `pip install -r requirements.txt`
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

## Observations

- **Classification (Wine)**: Logistic Regression excels (~0.9815) due to linear separability. SVM underperforms (~0.7593) with default parameters.
- **Classification (Breast Cancer)**: Random Forest is best (~0.9591), followed by Logistic Regression (~0.9474).
- **Clustering (Breast Cancer)**: K-Means (~0.3512) and Agglomerative Clustering (~0.3468) perform moderately; DBSCAN fails (~-0.2000) due to density issues. GMM (~0.3495) is comparable to K-Means.
- **Random State (K-Means)**: Fixing `random_state=42` ensures reproducibility, with minimal score variation.
- **Web App Features**: Toggle buttons allow viewers to show/hide classification and clustering observations from `observations.md` and `clustering_observations.md`.