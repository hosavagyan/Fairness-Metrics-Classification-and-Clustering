import numpy as np
import pandas as pd

from Fairness_Metric_Results.Fairness_Metrics import holistic_ai_metrics
from Fairness_Metric_Results.Model_Creation import train_test, model_pipline
from Fairness_Metric_Results import Fairness_Metrics as fm
from Fairness_Metric_Results.constants import acceptable_ranges, initial_groups

import warnings
warnings.filterwarnings('ignore')


def initial_setup(data_cls):

    target, df, sensitive_attr = data_cls.preprocess_adult()
    X_train, X_test, y_train, y_test = train_test(df, target)
    sensitive_attr = X_test[sensitive_attr]

    counterfacual_x = X_test.copy()

    counterfacual_x[data_cls.sensitive_attr] = counterfacual_x[data_cls.sensitive_attr].apply(lambda x: 1 - x)
    return X_train, X_test, y_train, y_test, sensitive_attr, counterfacual_x


def build_metics(X_train, X_test, y_train, y_test, sensitive_attr, counterfacual_x):
    final_df = pd.DataFrame(columns=['Metric', 'Value', 'Reference', 'Model'])
    bias_lst = []
    for i in model_pipline(X_train, X_test, y_train, y_test):
        handcoded_metrics = fm.test_metrics(model = i[0], counterfactual_x=counterfacual_x,
                                            y_pred=i[1], y_true=y_test, sensitive_attr=sensitive_attr,
                                            y_pred_proba=i[2])

        holistic_ai_metric_lst = holistic_ai_metrics(sensitive_attr=sensitive_attr, y_true=y_test,
                                                     y_pred=i[1], y_pred_proba=i[2])

        handcoded_metrics.extend(holistic_ai_metric_lst)

        metric_lst = [
            tuple(
                list(metric)+[i[3]])
            for metric in handcoded_metrics
        ]

        bias_lst.extend(metric_lst)

        fair_df = fm.holistic_ai_bulk(sensitive_attr=sensitive_attr, y_pred=i[1],
                                      y_true=y_test, X_test=X_test)

        fair_df['Model'] = i[3]
        final_df = pd.concat([final_df, fair_df], ignore_index=True)

    return final_df,bias_lst

def create_final_df(final_df, bias_lst):
    bias_df = pd.DataFrame(bias_lst, columns=['Metric', 'Value', 'Reference', 'Model'])
    final_df = pd.concat([final_df, bias_df], ignore_index=True).reset_index(drop=True)

    final_df['Acceptable Range'] = final_df['Metric'].apply(lambda x: acceptable_ranges[x])
    final_df['Normalized Differences'] = abs(final_df['Value'] - final_df['Reference']) / final_df['Acceptable Range']

    final_df['Group'] = np.nan
    for k, v in initial_groups.items():
        final_df['Group'].loc[final_df['Metric'].isin(v)] = k

    return final_df

def in_process_pipline(dataset):
    X_train, X_test, y_train, y_test, sensitive_attr, counterfacual_x = initial_setup(data_cls=dataset)

    final_df, bias_lst = build_metics(X_train, X_test, y_train,
                                      y_test, sensitive_attr, counterfacual_x)

    final_df = create_final_df(final_df, bias_lst)

    return final_df

