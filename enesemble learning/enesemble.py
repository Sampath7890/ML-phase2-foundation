from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
data = load_iris()

X = data.data
y = data.target


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create the models
logistic = LogisticRegression(max_iter=1000)

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

svm = SVC(
    probability=True,
    random_state=42
)


# Combine the models using voting
ensemble = VotingClassifier(
    estimators=[
        ("logistic", logistic),
        ("decision_tree", decision_tree),
        ("svm", svm)
    ],
    voting="hard"
)


# Train the models
logistic.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
svm.fit(X_train, y_train)

ensemble.fit(X_train, y_train)


# Make predictions
logistic_pred = logistic.predict(X_test)
tree_pred = decision_tree.predict(X_test)
svm_pred = svm.predict(X_test)
ensemble_pred = ensemble.predict(X_test)


# Check accuracy
print("Logistic Regression:", accuracy_score(y_test, logistic_pred))
print("Decision Tree:", accuracy_score(y_test, tree_pred))
print("SVM:", accuracy_score(y_test, svm_pred))
print("Ensemble:", accuracy_score(y_test, ensemble_pred))


# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    ensemble_pred,
    target_names=data.target_names
))


# Cross validation
scores = cross_val_score(
    ensemble,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("CV Scores:", scores)
print("Mean CV Score:", scores.mean())


# Predict for a new input
new_data = [[5.1, 3.5, 1.4, 0.2]]

prediction = ensemble.predict(new_data)

print("\nPredicted class:", prediction[0])
print("Predicted flower:", data.target_names[prediction[0]])