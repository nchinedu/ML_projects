from flask import Flask, render_template
from ml_models import load_and_split_data, train_logistic_regression, train_svm, train_decision_tree, train_random_forest

app = Flask(__name__)

@app.route('/')
def index():
    # Load data and train models
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    results = {
        'Logistic Regression': train_logistic_regression(X_train, y_train, X_test),
        'SVM': train_svm(X_train, y_train, X_test),
        'Decision Tree (random_state=42)': train_decision_tree(X_train, y_train, X_test, random_state=42),
        'Decision Tree (random_state=None)': train_decision_tree(X_train, y_train, X_test, random_state=None),
        'Random Forest': train_random_forest(X_train, y_train, X_test)
    }
    return render_template('index.html', results=results, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)