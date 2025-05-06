Wine and Breast Cancer Classification Web App
This project builds machine learning models to classify wine (using Logistic Regression, SVM, Decision Tree) and breast cancer data (adding Random Forest). Results are displayed in a Flask web app.
Setup

Clone the repository:
git clone <repository-url>
cd wine_breast_cancer_ml


Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Run the Flask app:
python app.py


Open http://localhost:5000 in a browser.


Project Structure

logistic_regression_wine.py: Logistic Regression on wine dataset.
svm_wine.py: SVM on wine dataset.
decision_tree_wine.py: Decision Tree on wine dataset.
breast_cancer_experiment.py: Experiment on breast cancer dataset.
app.py: Flask web app.
templates/index.html: HTML template.
requirements.txt: Dependencies.
README.md: Documentation.

Deployment to GitHub

Initialize a Git repository:
git init
git add .
git commit -m "Initial commit"


Push to GitHub:
git remote add origin <repository-url>
git push -u origin main


Deploy to Render:

Connect your GitHub repository.
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
Get the public URL (e.g., https://your-app.onrender.com).



Results
Wine Dataset

Logistic Regression: ~0.9815
SVM: ~0.7593
Decision Tree (random_state=42): ~0.9630
Decision Tree (random_state=None): ~0.9444 (varies)

Breast Cancer Dataset

Logistic Regression: ~0.9474
SVM: ~0.9123
Decision Tree (random_state=42): ~0.9181
Decision Tree (random_state=None): ~0.9064 (varies)
Random Forest: ~0.9591

Observations

Random State Effect (Wine Dataset): Setting random_state=42 in Decision Tree ensures reproducibility, yielding 0.9630 accuracy. Without it, accuracy varies (0.9444), showing the model’s sensitivity to random splits.
Wine Dataset: Logistic Regression performs best (0.9815), likely due to linear separability. SVM underperforms (0.7593) with default parameters.
Breast Cancer Dataset: Random Forest excels (0.9591) due to its ensemble approach, followed by Logistic Regression (0.9474).

