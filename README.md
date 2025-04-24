Breast Cancer Classification Web App
This project builds and compares machine learning models (Logistic Regression, SVM, Decision Tree, and Random Forest) for classifying breast cancer data using scikit-learn. The results are displayed in a Flask web app.
Setup

Clone the repository:
git clone <repository-url>
cd breast_cancer_ml


Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Run the Flask app:
python app.py


Open your browser and navigate to http://localhost:5000.


Project Structure

app.py: Flask web app.
ml_models.py: Machine learning model training and evaluation.
templates/index.html: HTML template for the web app.
requirements.txt: Project dependencies.
README.md: Project documentation.

Deployment to GitHub

Initialize a Git repository:
git init
git add .
git commit -m "Initial commit"


Create a repository on GitHub and push the code:
git remote add origin <repository-url>
git push -u origin main


Deploy to a hosting service (e.g., Render or Heroku) for a live web app:

Update app.py to bind to 0.0.0.0 and use the PORT environment variable.
Add a Procfile for the hosting service (e.g., web: gunicorn app:app).



Results

Logistic Regression: ~0.9474 accuracy
SVM: ~0.9123 accuracy
Decision Tree (random_state=42): ~0.9181 accuracy
Decision Tree (random_state=None): ~0.9064 accuracy (varies)
Random Forest: ~0.9591 accuracy

Observations

Setting random_state=42 in the Decision Tree ensures reproducible results, while random_state=None introduces variability.
Random Forest outperforms other models due to its ensemble nature.

