import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cabin_p1 = []
        self.cabin_p2 = []
        self.cabin_p3 = []

    def fit(self, X, y=None):
        self.cabin_p1 = list({i[0] for i in X[X["Pclass"] == 1]["Cabin"] if isinstance(i, str)})
        self.cabin_p2 = list({i[0] for i in X[X["Pclass"] == 2]["Cabin"] if isinstance(i, str)})
        self.cabin_p3 = list({i[0] for i in X[X["Pclass"] == 3]["Cabin"] if isinstance(i, str)})
        return self

    def transform(self, X):
        df = X.copy()
        def fill_cabin(row):
            num = np.random.randint(0, 99)
            if pd.isnull(row["Cabin"]) or isinstance(row["Cabin"], float):
                if row["Pclass"] == 1 and self.cabin_p1:
                    row["Cabin"] = np.random.choice(self.cabin_p1) + str(num)
                elif row["Pclass"] == 2 and self.cabin_p2:
                    row["Cabin"] = np.random.choice(self.cabin_p2) + str(num)
                elif row["Pclass"] == 3 and self.cabin_p3:
                    row["Cabin"] = np.random.choice(self.cabin_p3) + str(num)
                else:
                    row["Cabin"] = "U" + str(num)
            return row
        df = df.apply(fill_cabin, axis=1)
        df["Cabin"] = df["Cabin"].apply(lambda x: x[0] if isinstance(x, str) else "U")
        df["Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
        df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
        df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)
        df["FarePerPerson"] = df["Fare"] / df["Family_Size"]
        def age_group(age):
            if age < 13: return "Child"
            elif 13 <= age < 18: return "Teen"
            elif 18 <= age < 60: return "Adult"
            else: return "Senior"
        df["Age_Group"] = df["Age"].apply(age_group)
        df["Ticket_Frequency"] = df.groupby("Ticket")["Ticket"].transform("count")
        return df[["Pclass", "Sex", "Age", "Fare", "Embarked",
                   "Cabin", "Title", "Family_Size", "Is_Alone",
                   "Age_Group", "Ticket_Frequency"]]

# ---------------------------
class TitanicDataLoader:
    def __init__(self, path):
        self.path = path
    def load(self):
        df = pd.read_csv(self.path)
        df.set_index("PassengerId", inplace=True)
        df["Age"].fillna(df["Age"].mean(), inplace=True)
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
        return df