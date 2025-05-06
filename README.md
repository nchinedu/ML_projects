# Wine and Breast Cancer Classification Web App

This project builds machine learning models to classify wine (using Logistic Regression, SVM, Decision Tree) and breast cancer data (adding Random Forest). Results and detailed observations are displayed in a Flask web app, with an option to toggle the visibility of observations.

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

4. Open `http://localhost:5000` in a browser. Click "Show Observations" to view detailed analysis.

## Project Structure

- `logistic_regression_wine.py`: Logistic Regression on wine dataset.
- `svm_wine.py`: SVM on wine dataset.
- `decision_tree_wine.py`: Decision Tree on wine dataset.
- `breast_cancer_experiment.py`: Experiment on breast cancer dataset.
- `app.py`: Flask web app.
- `templates/index.html`: HTML template.
- `observations.md`: Detailed observations and answers.
- `requirements.txt`: Dependencies.
- `README.md`: Documentation.

## Deployment to GitHub

1. Initialize a Git repository:
   ```bash
   git init
   git add .
   git commit -m "Add observations and toggle feature"
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

### Wine Dataset
- Logistic Regression: ~0.9815
- SVM: ~0.7593
- Decision Tree (random_state=42): ~0.9630
- Decision Tree (random_state=None): ~0.9444 (varies)

### Breast Cancer Dataset
- Logistic Regression: ~0.9474
- SVM: ~0.9123
- Decision Tree (random_state=42): ~0.9181
- Decision Tree (random_state=None): ~0.9064 (varies)
- Random Forest: ~0.9591

## Observations

- **Random State Effect (Wine Dataset)**: Setting `random_state=42` in Decision Tree ensures reproducibility (~0.9630 accuracy). Without it, accuracy varies (~0.9444), showing sensitivity to random splits.
- **Wine Dataset**: Logistic Regression performs best (~0.9815) due to linear separability. SVM underperforms (~0.7593) with default parameters.
- **Breast Cancer Dataset**: Random Forest excels (~0.9591) due to its ensemble approach, followed by Logistic Regression (~0.9474).
- **Web App Feature**: Viewers can toggle observations via a "Show Observations" button, displaying detailed analysis from `observations.md`.