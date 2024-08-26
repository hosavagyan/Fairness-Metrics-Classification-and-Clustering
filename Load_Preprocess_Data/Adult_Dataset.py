from holisticai.datasets import load_adult
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Adult:
    def __init__(self, sensitive_attr):
        self.sensitive_attr = sensitive_attr

    @staticmethod
    def _load_data():
        return load_adult()['frame']

    @staticmethod
    def _clean_df(df: pd.DataFrame):
        df = df.dropna()
        df['class'] = df['class'].apply(lambda x: 1 if x == ">50K" else 0)
        df['race'] = df['race'].apply(lambda x: 1 if x.lower() == 'white' else 0)
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame):
        # df['education'] = df['education'].apply(lambda x: Adult.education_dict[x])
        df['age-hours'] = df['age'] * df['hours-per-week']

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
        numerical_cols = df.loc[:, ~df.columns.isin(['class', 'sex', 'race'])].columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def preprocess_adult(self):
        df = self._load_data()
        df = self._clean_df(df)
        df = self._feature_engineering(df)
        df = self._encoding(df)
        df = self._scaling(df)

        drop_elements = ['education', 'native-country', 'class']
        target = df["class"]
        df = df.drop(drop_elements, axis=1)

        return target,df, self.sensitive_attr