import pandas as pd     
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Salary_Data.csv")
df.head()
df.shape
plt.figure(figsize=(8,5))
sns.scatterplot(x='Years of Experience', y='Salary', data=df)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show
from sklearn.preprocessing import LabelEncoder

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Education Level"] = le_education.fit_transform(df["Education Level"])
df["Job Title"] = le_job.fit_transform(df["Job Title"])
X = df.drop("Salary", axis=1)
y = df["Salary"]
from sklearn.model_selection import train_test_split

# Features and Target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Split the data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X_test)
# Predictions
y_pred = model.predict(X_test)
y_pred[0:10]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
data = [[
    30,    # Age
    1,     # Gender (Male)
    0,     # Education Level (Bachelor)
    15,    # Job Title (encoded value)
    5      # Years of Experience
]]

prediction = model.predict(data)
print("Predicted Salary:", prediction)