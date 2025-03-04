import biasondemand
from itertools import combinations
import pandas as pd

def generate_ds(param_dict):
    print("Start creation dataset.", '\n')
    biasondemand.generate_dataset(**param_dict)
def generate_article_datasets():
    all_dicts = {}
    all_dicts_num = {}

    base_path = "datasets/combinations/biased/"
    dim = 10000

    # Define the bias types and their ranges
    bias_types = {
        "l_y": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on Y
        "l_yr": [0.1, 1.5, 3, 6, 9],
        "l_m_y": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Measurement bias on Y
        "l_m_yr": [0.1, 1.5, 3, 6, 9],  # Measurement bias on Y
        "l_h_r": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on R
        "l_h_rr": [0.1, 1.5, 3, 6, 9],  # Hist bias on R reduced
        "l_h_q": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on Q
        "l_h_qr": [0.1, 1.5, 3, 6, 9],
        "l_m": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Measurement bias on R
        "l_mr": [0.1, 1.5, 3, 6, 9],  # Measurement bias on R
        "p_u": [0.003, 0.006, 0.008, 0.01, 0.1, 0.3, 0.5], # Sample bias [0.5, 0.3, 0.1, 0.01, 0.008, 0.006, 0.003]
        "p_ur": [0.003, 0.008, 0.1, 0.5],  # Sample bias [0.5, 0.1, 0.008, 0.003]
        "p_u_r": [0.3, 0.5, 0.7, 0.9], # Sample bias for representation 0.9, 0.7, 0.5, 0.3
        "l_r": [True, False], # Representation bias
        "l_o": [True, False], # Omission
        "l_y_b": [0] # Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R
    }

    # measurement on R
    def match_l_m(val):
        if val in [0.1, 0.5, 1]:
            return 'Low'
        elif val in [1.5, 2, 3, 4, 5]:
            return 'Med'
        else:
            return 'High'

    def match_p_u(val):
        if val in [0.5, 0.3, 0.1]:
            return 'High'
        else:
            return 'Low'

    def match_p_u_r(val):
        if val in [0.9, 0.7]:
            return 'High'
        else:
            return 'Low'

    # S1 - no historical bias
    # no additional bias
    all_dicts['S1A'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 1, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
    all_dicts_num["S1A"] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}

    # measurment on R
    for v in bias_types["l_m"]:
        ind = bias_types["l_m"].index(v)
        val = match_l_m(v)
        all_dicts[f'S1B{ind}'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S1B{ind}']['B-{val}'.format(val=val)] = 1

        all_dicts_num[f'S1B{ind}'] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S1B{ind}']['B'] = v


    # omission
    all_dicts['S1C'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 1, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}

    all_dicts_num[f'S1C'] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 1, "D": 0, "E": 0, "F": 0}

    # sample
    for v in bias_types["p_u"]:
        ind = bias_types["p_u"].index(v)
        val = match_p_u(v)
        all_dicts[f'S1D{ind}'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S1D{ind}']['D-{val}'.format(val=val)] = 1

        all_dicts_num[f'S1D{ind}'] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S1D{ind}']['D'] = v

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_m_y"]:
        ind = bias_types["l_m_y"].index(v)
        val = match_l_m(v)
        all_dicts[f'S1E{ind}'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0,
                                  "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                                  "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                                  "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S1E{ind}']['E-{val}'.format(val=val)] = 1

        all_dicts_num[f'S1E{ind}'] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,"F": 0}
        all_dicts_num[f'S1E{ind}']['E'] = v

    # representation bias p_u(variable)
    for v in bias_types["p_u_r"]:
        ind = bias_types["p_u_r"].index(v)
        val = match_p_u_r(v)
        all_dicts[f'S1F{ind}'] = {"S1": 1, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0,
                                  "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                                  "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                                  "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S1F{ind}']['F-{val}'.format(val=val)] = 1

        all_dicts_num[f'S1F{ind}'] = {"S1": 1, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S1F{ind}']['F'] = v

    # S2 historical bias on R
    # no additional bias
    for v in bias_types["l_h_r"]:
        ind = bias_types["l_h_r"].index(v)
        val = match_l_m(v)
        all_dicts[f'S2A{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 1, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S2A{ind}']['S2-{val}'.format(val=val)] = 1

        all_dicts_num[f'S2A{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S2A{ind}']['S2'] = v



    # measurment on R
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_mr"]:
            ind2 = bias_types["l_mr"].index(v2)
            val2 = match_l_m(v2)
            all_dicts[f'S2B{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S2B{ind}_{ind2}']['S2-{val}'.format(val=val)] = 1
            all_dicts[f'S2B{ind}_{ind2}']['B-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S2B{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S2B{ind}_{ind2}']['S2'] = v
            all_dicts_num[f'S2B{ind}_{ind2}']['B'] = v2


    # omission
    for v in bias_types["l_h_r"]:
        ind = bias_types["l_h_r"].index(v)
        val = match_l_m(v)
        all_dicts[f'S2C{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 1, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S2C{ind}']['S2-{val}'.format(val=val)] = 1

        all_dicts_num[f'S2C{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 1, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S2C{ind}']['S2'] = 1

    # sample
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            val2 = match_p_u(v2)
            all_dicts[f'S2D{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S2D{ind}_{ind2}']['S2-{val}'.format(val=val)] = 1
            all_dicts[f'S2D{ind}_{ind2}']['D-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S2D{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S2D{ind}_{ind2}']['S2'] = v
            all_dicts_num[f'S2D{ind}_{ind2}']['D'] = v2

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            val2 = match_l_m(v2)
            all_dicts[f'S2E{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S2E{ind}_{ind2}']['S2-{val}'.format(val=val)] = 1
            all_dicts[f'S2E{ind}_{ind2}']['E-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S2E{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S2E{ind}_{ind2}']['S2'] = v
            all_dicts_num[f'S2E{ind}_{ind2}']['E'] = v2


    # representation bias p_u
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            val2 = match_p_u_r(v2)
            all_dicts[f'S2F{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S2F{ind}_{ind2}']['S2-{val}'.format(val=val)] = 1
            all_dicts[f'S2F{ind}_{ind2}']['F-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S2F{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S2F{ind}_{ind2}']['S2'] = v
            all_dicts_num[f'S2F{ind}_{ind2}']['F'] = v2


    # S3 - Historical bias on Y
    # no additional bias
    for v in bias_types["l_y"]:
        ind = bias_types["l_y"].index(v)
        val = match_l_m(v)
        all_dicts[f'S3A{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 1, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S3A{ind}']['S3-{val}'.format(val=val)] = 1

        all_dicts_num[f'S3A{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S3A{ind}']['S3'] = v

    # measurment on R
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_mr"]:
            val2 = match_l_m(v2)
            ind2 = bias_types["l_mr"].index(v2)
            all_dicts[f'S3B{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S3B{ind}_{ind2}']['S3-{val}'.format(val=val)] = 1
            all_dicts[f'S3B{ind}_{ind2}']['B-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S3B{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S3B{ind}_{ind2}']['S3'] = v
            all_dicts_num[f'S3B{ind}_{ind2}']['B'] = v2

    # omission
    for v in bias_types["l_y"]:
        ind = bias_types["l_y"].index(v)
        val = match_l_m(v)
        all_dicts[f'S3C{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 1, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S3C{ind}']['S3-{val}'.format(val=val)] = 1

        all_dicts_num[f'S3C{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 1, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S3C{ind}']['S3'] = v2

    # sample
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            val2 = match_p_u(v2)
            all_dicts[f'S3D{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S3D{ind}_{ind2}']['S3-{val}'.format(val=val)] = 1
            all_dicts[f'S3D{ind}_{ind2}']['D-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S3D{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S3D{ind}_{ind2}']['S3'] = v
            all_dicts_num[f'S3D{ind}_{ind2}']['D'] = v2


    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            val2 = match_l_m(v2)
            all_dicts[f'S3E{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S3E{ind}_{ind2}']['S3-{val}'.format(val=val)] = 1
            all_dicts[f'S3E{ind}_{ind2}']['E-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S3E{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S3E{ind}_{ind2}']['S3'] = v
            all_dicts_num[f'S3E{ind}_{ind2}']['E'] = v2

    # representation bias p_u
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            val2 = match_p_u_r(v2)
            all_dicts[f'S3F{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S3F{ind}_{ind2}']['S3-{val}'.format(val=val)] = 1
            all_dicts[f'S3F{ind}_{ind2}']['F-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S3F{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S3F{ind}_{ind2}']['S3'] = v
            all_dicts_num[f'S3F{ind}_{ind2}']['F'] = v2


    # S4 - historical bias on Q
    # no additional bias
    for v in bias_types["l_h_q"]:
        ind = bias_types["l_h_q"].index(v)
        val = match_l_m(v)
        all_dicts[f'S4A{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 1, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S4A{ind}']['S4-{val}'.format(val=val)] = 1

        all_dicts_num[f'S4A{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S4A{ind}']['S4'] = v

    # measurment on R
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_mr"]:
            ind2 = bias_types["l_mr"].index(v2)
            val2 = match_l_m(v2)
            all_dicts[f'S4B{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S4B{ind}_{ind2}']['S4-{val}'.format(val=val)] = 1
            all_dicts[f'S4B{ind}_{ind2}']['B-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S4B{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0,"F": 0}
            all_dicts_num[f'S4B{ind}_{ind2}']['S4'] = v
            all_dicts_num[f'S4B{ind}_{ind2}']['B'] = v2

    # omission
    for v in bias_types["l_h_q"]:
        ind = bias_types["l_h_q"].index(v)
        val = match_l_m(v)
        all_dicts[f'S4C{ind}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 1, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
        all_dicts[f'S4C{ind}']['S4-{val}'.format(val=val)] = 1

        all_dicts_num[f'S4C{ind}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 1, "D": 0, "E": 0, "F": 0}
        all_dicts_num[f'S4C{ind}']['S4'] = v

    # sample
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            val2 = match_p_u(v2)
            all_dicts[f'S4D{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S4D{ind}_{ind2}']['S4-{val}'.format(val=val)] = 1
            all_dicts[f'S4D{ind}_{ind2}']['D-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S4D{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S4D{ind}_{ind2}']['S4'] = v
            all_dicts_num[f'S4D{ind}_{ind2}']['D'] = v2

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            val2 = match_l_m(v2)
            all_dicts[f'S4E{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S4E{ind}_{ind2}']['S4-{val}'.format(val=val)] = 1
            all_dicts[f'S4E{ind}_{ind2}']['E-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S4E{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S4E{ind}_{ind2}']['S4'] = v
            all_dicts_num[f'S4E{ind}_{ind2}']['E'] = v2


    # representation bias p_u
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        val = match_l_m(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            val2 = match_p_u_r(v2)
            all_dicts[f'S4F{ind}_{ind2}'] = {"S1": 0, "S2-Low": 0, "S2-Med": 0, "S2-High": 0, "S3-Low": 0, "S3-Med": 0, "S3-High": 0, "S4-Low": 0, "S4-Med": 0, "S4-High": 0,
                        "A": 0, "B-Low": 0, "B-Med": 0, "B-High": 0, "C": 0, "D-Low": 0, "D-High": 0,
                        "E-Low": 0, "E-Med": 0, "E-High": 0, "F-Low": 0, "F-High": 0}
            all_dicts[f'S4F{ind}_{ind2}']['S4-{val}'.format(val=val)] = 1
            all_dicts[f'S4F{ind}_{ind2}']['F-{val}'.format(val=val2)] = 1

            all_dicts_num[f'S4F{ind}_{ind2}'] = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
            all_dicts_num[f'S4F{ind}_{ind2}']['S4'] = v
            all_dicts_num[f'S4F{ind}_{ind2}']['F'] = v2

    return all_dicts, all_dicts_num


if __name__ == "__main__":
    data, data_num = generate_article_datasets()

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    df_num = pd.DataFrame(data_num)

    # Transpose the DataFrame to match the structure
    df = df.T
    df_num = df_num.T

    # Print the DataFrame
    df.to_csv("datasets/analysis/df_properties.csv")
    df_num.to_csv("datasets/analysis/df_properties_numbered.csv")
