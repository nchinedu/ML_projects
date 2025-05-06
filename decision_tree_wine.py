from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

<<<<<<< HEAD
def train_decision_tree_wine(random_state=50):
=======
def train_decision_tree_wine(random_state=42):
>>>>>>> 90d57f96f2429b621e516b5a8e4e8e7472efe4fe
    # Load wine dataset
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split data
<<<<<<< HEAD
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
=======
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
>>>>>>> 90d57f96f2429b621e516b5a8e4e8e7472efe4fe
    
    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, feature_names

if __name__ == "__main__":
    accuracy_42, _ = train_decision_tree_wine(random_state=42)
    accuracy_none, _ = train_decision_tree_wine(random_state=None)
<<<<<<< HEAD
    print(f"Decision Tree Accuracy (random_state=50) on Wine Dataset: {accuracy_42:.4f}")
=======
    print(f"Decision Tree Accuracy (random_state=42) on Wine Dataset: {accuracy_42:.4f}")
>>>>>>> 90d57f96f2429b621e516b5a8e4e8e7472efe4fe
    print(f"Decision Tree Accuracy (random_state=None) on Wine Dataset: {accuracy_none:.4f}")