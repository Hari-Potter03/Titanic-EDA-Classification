# 📘 Study Plan: Titanic EDA & Classification

## 🎯 Objective
To practice end-to-end supervised learning by predicting survival on the Titanic. Focus on:
- Data wrangling (cleaning, handling missing values)
- Exploratory data analysis (EDA)
- Feature engineering
- Classification modeling (Logistic Regression)
- Model evaluation and visualization

---

## 🔍 Skills Practiced
✅ EDA + Visualization (Round 1, Round 2)  
✅ Data wrangling and curation (Technical Screen)  
✅ Regression/classification modeling (Technical Screen, Round 2)  
✅ Communication of insights (Round 2)  
✅ End-to-end workflow with Pandas, Scikit-learn, Matplotlib  

---

## 🗓️ 3-Day Task Breakdown

### Day 1 – EDA & Cleaning
- Load dataset and explore structure with `.info()`, `.describe()`
- Visualize survival by `Sex`, `Pclass`, `Embarked`
- Handle missing values (`Age`, `Embarked`)
- Plot distributions for `Age`, `Fare`, heatmap for missingness

### Day 2 – Feature Engineering & Modeling
- Create new features: `FamilySize`, `IsAlone`, `Title`
- Encode categorical vars
- Split into train/test and fit a `LogisticRegression` model
- Build a pipeline with preprocessing steps

### Day 3 – Evaluation & Communication
- Evaluate with accuracy, precision, recall, F1, ROC
- Plot confusion matrix, feature importances
- Export visuals to `/outputs/figures`
- Summarize findings in `summary.md`

---
