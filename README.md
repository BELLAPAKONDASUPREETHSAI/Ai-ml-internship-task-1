# titanic_cleaning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Data loaded successfully!")

# 2. Explore basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

# 4. Encode categorical features
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 5. Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 6. Outlier detection and removal using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers(df, 'Fare')

# 7. Boxplot to visualize outliers
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age and Fare after cleaning")
plt.savefig("boxplot.png")
plt.show()

# 8. Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)
print("Data cleaned and saved to 'cleaned_titanic.csv'")
