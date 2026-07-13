# Import all libraries at the top
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load dataset
df = pd.read_csv("loan_prediction_dataset.csv")

df.head()
df.info()

# Encode Employment_Status
le = LabelEncoder()
df['Employment_Status'] = le.fit_transform(df['Employment_Status'])
print(le.classes_)

# Feature selection
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Training complete")

# Make predictions
y_pred = model.predict(X_test)
print(y_pred[:10])

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Reject", "Approve"],
    filled=True
)
plt.show()