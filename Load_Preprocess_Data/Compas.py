from holisticai.datasets import load_adult
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class COMPAS:
    def __init__(self, sensitive_attr):
        self.sensitive_attr = sensitive_attr

    @staticmethod
    def _load_data():
        return pd.read_csv(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        )

    @staticmethod
    def _clean_df(df: pd.DataFrame):
        to_keep = ['sex', 'age', 'race', 'decile_score', 'priors_count',
       'c_days_from_compas', 'c_charge_degree', 'is_violent_recid',
       'v_decile_score', 'two_year_recid', 'c_jail_in', 'c_jail_out']

        df = df[to_keep]
        return df

    @staticmethod
    def _feature_engineering(df: pd.DataFrame):
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['days_in_jail'] = abs(
            (df['c_jail_out'] - df['c_jail_in']).dt.days
        )
        df['age'] = df['age'].apply(
            lambda x: 1 if x < 30 else 0
        )

        numeric_cols = ['c_days_from_compas', 'v_decile_score']
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        categorical_cols = ['days_in_jail', 'c_charge_degree', 'race', 'sex' ]
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

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
        scaler = StandardScaler()
        numerical_cols = df.loc[:, ~df.columns.isin(
            ['age', 'sex', 'two_year_recid','c_charge_degree',
             'is_violent_recid', 'c_jail_in', 'c_jail_out']
        )].columns

        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def preprocess_adult(self):
        df = self._load_data()
        df = self._clean_df(df)
        df = self._feature_engineering(df)
        df = self._encoding(df)
        df = self._scaling(df)

        drop_elements = ['two_year_recid','c_jail_in', 'c_jail_out']
        target = df["two_year_recid"]
        df = df.drop(drop_elements, axis=1)

        return target,df, self.sensitive_attr