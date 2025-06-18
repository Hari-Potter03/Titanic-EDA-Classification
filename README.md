# Titanic Survival Prediction: EDA & Logistic Regression

Predicting passenger survival using exploratory data analysis and logistic regression.

---

## 🚀 Project Goals
- Clean and explore the Titanic dataset
- Engineer new features to improve prediction accuracy
- Train and evaluate a logistic regression classifier

---

## 📦 Dataset
- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)
- Fields: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Survived`, etc.

---

## 📌 Coding Exercises

### 1. Exploratory Data Analysis (EDA)
- Calculate survival rate by gender, class, and embarkation point
- Plot distributions for `Age`, `Fare`, and `Pclass`
- Identify and visualize missing values (use seaborn heatmap)

### 2. Feature Engineering
- Create new columns:
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = 1 if FamilySize == 1 else 0`
  - Extract `Title` from the `Name` column
- Encode `Sex`, `Embarked`, and `Title` using One-Hot Encoding

### 3. Model Training
- Split into train/test
- Build a logistic regression model using `scikit-learn`
- Evaluate with accuracy, precision, recall, and confusion matrix
- Plot ROC curve using `sklearn.metrics`

### 4. Bonus
- Compare performance with `RandomForestClassifier`
- Use `Pipeline` and `ColumnTransformer` for a clean workflow

---

## 🧠 Key Questions
- Which features most influence survival?
- Does adding `IsAlone` improve performance?
- What happens when you drop low-importance variables?

---
