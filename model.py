import pickle
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Fetch dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train classifier
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

# Print model accuracy
print(f"Model accuracy: {clf.score(X_test, y_test)}")

# Save model
with open('mnist_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
