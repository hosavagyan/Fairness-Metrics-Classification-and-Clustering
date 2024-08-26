import numpy as np
import pandas as pd
from holisticai.bias.metrics import abroca, accuracy_diff, z_test_diff, z_test_ratio, classification_bias_metrics
from sklearn.metrics import confusion_matrix, accuracy_score, adjusted_mutual_info_score

# fixme MAKE THIS WHOLE THING CLASS STRUCTURE

def demographic_parity(y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    group_parity = {}
    for group in unique_groups:
        group_indices = (sensitive_attr == group)
        group_parity[group] = np.mean(y_pred[group_indices])
    return demographic_parity.__name__, max(group_parity.values()), 0

def mean_difference(y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    group_means = {}
    for group in unique_groups:
        group_indices = (sensitive_attr == group)
        group_means[group] = np.mean(y_pred[group_indices])
    mean_diff = max(group_means.values()) - min(group_means.values())
    return mean_difference.__name__, mean_diff, 0


def equal_opportunity(y_true, y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    group_tpr = {}
    for group in unique_groups:
        group_indices = (sensitive_attr == group)
        tn, fp, fn, tp = confusion_matrix(y_true[group_indices], y_pred[group_indices]).ravel()
        group_tpr[group] = tp / (tp + fn) if (tp + fn) > 0 else 0
    return equal_opportunity.__name__, max(group_tpr.values()), 0

def predictive_equality(y_true, y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    group_fpr = {}
    for group in unique_groups:
        group_indices = (sensitive_attr == group)
        tn, fp, fn, tp = confusion_matrix(y_true[group_indices], y_pred[group_indices]).ravel()
        group_fpr[group] = fp / (fp + tn) if (fp + tn) > 0 else 0
    return predictive_equality.__name__, max(group_fpr.values()), 0


def predictive_parity(y_true, y_pred, sensitive_attr):
    unique_groups = np.unique(sensitive_attr)
    group_ppv = {}
    for group in unique_groups:
        group_indices = (sensitive_attr == group)
        tn, fp, fn, tp = confusion_matrix(y_true[group_indices], y_pred[group_indices]).ravel()
        group_ppv[group] = tp / (tp + fp) if (tp + fp) > 0 else 0
    return predictive_parity.__name__, max(group_ppv.values()), 0


def mutual_info(sensitive_attr, y_pred):
    mutual_info_score = adjusted_mutual_info_score(sensitive_attr,y_pred)
    return mutual_info.__name__, mutual_info_score, 0

def counterfactual_fairness(model, counterfactual_x, y_true, y_pred, y_pred_proba):
    counterfactual_y_pred = model.predict(counterfactual_x)
    counter_f = ('counterfactual_fairness', np.mean(y_pred == counterfactual_y_pred), 1)

    acc_non_counterf = accuracy_score(y_true, y_pred)
    acc_counterf = accuracy_score(y_true, counterfactual_y_pred)
    acc_diff= abs(acc_non_counterf - acc_counterf)
    counter_acc = ('counterfactual_accuracy', acc_diff, 0)

    y_pred_probs = y_pred_proba[:, 1]
    counterfactual_y_pred = model.predict_proba(counterfactual_x)[:, 1]
    counter_cons = ('counterfactual_consistency', np.mean(np.abs(y_pred_probs - counterfactual_y_pred)), 0)

    return counter_f, counter_acc, counter_cons

# def yield_metrics():
#     fairness_list = [demographic_parity, mean_difference, equal_opportunity,
#                      predictive_equality, predictive_parity, counterfactual_fairness]
#     return (metric for metric in fairness_list)


def test_metrics(model, counterfactual_x, y_pred,y_true, sensitive_attr, y_pred_proba):
    d_p = demographic_parity(y_pred=y_pred, sensitive_attr=sensitive_attr)

    m_d = mean_difference(y_pred=y_pred, sensitive_attr=sensitive_attr)

    e_o = equal_opportunity(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)

    pred_eq = predictive_equality(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)

    pred_p = predictive_parity(y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive_attr)

    mut_inf = mutual_info(sensitive_attr=sensitive_attr, y_pred=y_pred)

    c_f,c_a,c_c = counterfactual_fairness(model=model, counterfactual_x=counterfactual_x,
                                      y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)

    lst = [d_p,m_d,e_o,pred_eq,pred_p,mut_inf,c_f,c_a,c_c]

    return lst

def holistic_ai_metrics(sensitive_attr,y_true, y_pred, y_pred_proba):
    group_a = (sensitive_attr == 1).astype(int)
    group_b = (sensitive_attr == 0).astype(int)

    abroca_value = abroca(group_a, group_b, y_pred_proba[:, 1], y_true)
    abroca_tup = ('abroca', abroca_value, 0)

    acc_diff = accuracy_diff(group_a, group_b, y_pred, y_true)
    acc_diff_tup = ('acc_diff', acc_diff, 0)

    ztest_diff = z_test_diff(group_a,group_b,y_pred)
    ztest_diff_tup = ('z_test_diff', ztest_diff, 0)

    ztest_ratio = z_test_ratio(group_a,group_b,y_pred)
    ztest_ratio_tup = ('z_test_ratio', ztest_ratio, 0)

    lst = [abroca_tup, acc_diff_tup, ztest_diff_tup, ztest_ratio_tup]

    return lst

def holistic_ai_bulk(sensitive_attr, y_pred, y_true,X_test):
    group_a = (sensitive_attr == 1).astype(int)
    group_b = (sensitive_attr == 0).astype(int)

    equal_outcome = classification_bias_metrics(group_a, group_b, y_pred, y_true,
                                    metric_type='equal_outcome').reset_index()
    equal_opp = classification_bias_metrics(group_a, group_b, y_pred, y_true,
                                    metric_type='equal_opportunity').reset_index()
    individual = classification_bias_metrics(y_pred=y_pred, y_true=y_true, X=X_test,
                                    metric_type='individual').reset_index()

    df = pd.concat([equal_outcome,equal_opp], ignore_index =True)
    df = pd.concat([df, individual], ignore_index=True).iloc[[1,2,3,7,9]]

    return df