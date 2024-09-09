from holisticai.datasets import load_adult
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Bank:
    def __init__(self, sensitive_attr):
        self.sensitive_attr = sensitive_attr

    @staticmethod
    def _load_data():
        return pd.read_csv(
            r'.\Load_Preprocess_Data\bank-additional-full.csv',
            sep=';'
        )

    @staticmethod
    def _clean_df(df: pd.DataFrame):
        to_drop = ['contact', 'day_of_week', 'duration', 'month',
                   'pdays', 'nr.employed', 'euribor3m', 'emp.var.rate']
        df = df.drop(to_drop, axis = 1)
        df = df.dropna()
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame):
        df['marital'] = df['marital'].apply(
            lambda x: 1 if x == 'married' else 0
        )

        df['age'] = df['age'].apply(
            lambda x: 1 if x < 40 else 0
        )

        return df

    @staticmethod
    def _encoding(df: pd.DataFrame):
        # Label Encoding categorical columns
        label_encoders = {}
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype == 'category':
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])
                label_encoders[column] = le
        return df

    @staticmethod
    def _scaling(df: pd.DataFrame):
        # Scaling numerical columns
        scaler = StandardScaler()
        numerical_cols = df.loc[:, ~df.columns.isin(['age', 'marital', 'y'])].columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def preprocess_adult(self):
        df = self._load_data()
        df = self._clean_df(df)
        df = self._feature_engineering(df)
        df = self._encoding(df)
        df = self._scaling(df)

        drop_elements = ['y']
        target = df["y"]
        df = df.drop(drop_elements, axis=1)

        return target,df, self.sensitive_attr