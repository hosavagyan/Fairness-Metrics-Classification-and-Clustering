from Load_Preprocess_Data.Adult_Dataset import Adult
from Load_Preprocess_Data.Bank_dataset import Bank
from Load_Preprocess_Data.Compas import COMPAS

acceptable_ranges = {
    'counterfactual_fairness': 0.05, # handwritten
    'counterfactual_accuracy': 0.05, # handwritten
    'counterfactual_consistency': 0.05, # handwritten
    'Four Fifths Rule': 0.2,  # holisticAI  ✔ Normalized to [0.8, 1]
    'mean_difference': 0.2, # handwritten ✔
    'Theil Index': 0.3,  # holisticAI ✔
    'Disparate Impact': 0.4,  # holisticAI  ✔ Normalized to [0.8, 1.2]
    'Cohen D': 0.5, # holisticAI  ✔ (0.2, 0.5, 0.8)
    'predictive_equality': 0.2, # handwritten ✔
    'Average Odds Difference': 0.2, # holisticAI  ✔
    'equal_opportunity': 0.2, # handwritten ✔
    'predictive_parity': 0.2, # handwritten ✔
    'abroca': 0.4, # holisticAI,  range(-1,1) ✔
    'acc_diff': 0.4, # holisticAI,  range(-1,1) ✔
    'mutual_info': 0.2, # sklearn, no universal accepted -> close to 0 =>  ✔
    'z_test_diff': 4,   # from -2 to 2 is good  holisticAI  ✔
    'z_test_ratio': 4  # holisticAI  ✔
}

initial_groups = {
    1: ['Disparate Impact', 'Four Fifths Rule',
        'mean_difference', 'Theil Index'],
    2: ['Average Odds Difference', 'predictive_equality', 'equal_opportunity',
        'abroca', 'predictive_parity', 'acc_diff', 'z_test_diff', 'z_test_ratio'],
    3: ['counterfactual_fairness', 'counterfactual_accuracy', 'counterfactual_consistency',
        'Cohen D', 'mutual_info']
}

lst_of_data = [(Adult, 'race'), (Adult, 'sex'), (Bank, 'age'),
               (Bank,'marital'), (COMPAS, 'age'),(COMPAS, 'sex')]
