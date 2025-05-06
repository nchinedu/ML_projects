from flask import Flask, render_template, request
from logistic_regression_wine import train_logistic_regression_wine
from svm_wine import train_svm_wine
from decision_tree_wine import train_decision_tree_wine
from breast_cancer_experiment import (
    load_and_split_breast_cancer,
    train_logistic_regression_bc,
    train_svm_bc,
    train_decision_tree_bc,
    train_random_forest_bc
)
import markdown
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Wine dataset results
    wine_results = {
        'Logistic Regression': train_logistic_regression_wine()[0],
        'SVM': train_svm_wine()[0],
        'Decision Tree (random_state=42)': train_decision_tree_wine(random_state=42)[0],
        'Decision Tree (random_state=None)': train_decision_tree_wine(random_state=None)[0]
    }
    wine_feature_names = train_logistic_regression_wine()[1]
    
    # Breast cancer dataset results
    X_train, X_test, y_train, y_test, bc_feature_names = load_and_split_breast_cancer()
    bc_results = {
        'Logistic Regression': train_logistic_regression_bc(X_train, y_train, X_test, y_test),
        'SVM': train_svm_bc(X_train, y_train, X_test, y_test),
        'Decision Tree (random_state=42)': train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=42),
        'Decision Tree (random_state=None)': train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=None),
        'Random Forest': train_random_forest_bc(X_train, y_train, X_test, y_test)
    }
    
    # Read observations
    show_observations = request.args.get('show_observations', 'false').lower() == 'true'
    observations_html = ''
    if show_observations:
        with open('observations.md', 'r') as f:
            observations_md = f.read()
            observations_html = markdown.markdown(observations_md)
    
    return render_template('index.html', 
                         wine_results=wine_results, 
                         bc_results=bc_results, 
                         wine_feature_names=wine_feature_names, 
                         bc_feature_names=bc_feature_names,
                         show_observations=show_observations,
                         observations_html=observations_html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)