import biasondemand
from itertools import combinations

def generate_unbiased_dataset():
    path = "datasets/unbiased/unbiased_ds"
    biasondemand.generate_dataset(
        path=path,
        dim=1000,
        sy=0.0,  # No noise for unbiased dataset
        l_q=0.0,  # No importance of Q for Y
        l_r_q=0.0,  # No influence from R to Q
        thr_supp=1.0  # No threshold for discarding features
    )
    print(f"Unbiased dataset generated at {path}")

def generate_biased_datasets():
    base_path = "datasets/biased/"
    dim = 1000

    # Define the bias types and their ranges
    bias_types = {
        "l_y": [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on Y
        "l_m_y": [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Measurement bias on Y
        "l_h_r": [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on R
        "l_h_q": [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Hist bias on Q
        "l_m": [0, 0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9], # Measurement bias on R
        "p_u": [0.01, 0.008, 0.006, 0.003 , 0.0002, 0.0001, 0.00009], # Sample bias
        "l_r": [True, False], # Representation bias
        "l_o": [True, False], # Omission
        "l_y_b": [0] # Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R
    }

    # dim=15000, l_y=4, l_m_y=0, thr_supp=1, l_h_r=1.5,  l_h_q=1, l_m=1, p_u=1, l_r=False, l_o=False, l_y_b=0, l_q=2, sy=5, l_r_q=0, l_m_y_non_linear=False
    # Variability parameters
    sy_values = [0.5]
    l_q_values = [0.5]
    l_r_q_values = [0.5]
    thr_supp_values = [1]

    # Generate datasets for single biases
    for bias_type, values in bias_types.items():
        for value in values:
            if isinstance(value, bool):
                value_str = "true" if value else "false"
            else:
                value_str = str(value).replace('.', '_')
            for sy in sy_values:
                for l_q in l_q_values:
                    for l_r_q in l_r_q_values:
                        for thr_supp in thr_supp_values:
                            path = f"{base_path}{bias_type}_{value_str}_sy_{sy}_lq_{l_q}_lrq_{l_r_q}_thrsupp_{thr_supp}"
                            kwargs = {
                                "path": path,
                                "dim": dim,
                                "sy": sy,
                                "l_q": l_q,
                                "l_r_q": l_r_q,
                                "thr_supp": thr_supp
                            }
                            kwargs[bias_type] = value
                            try:
                                biasondemand.generate_dataset(**kwargs)
                                print(f"Biased dataset generated at {path} with {bias_type}={value}, sy={sy}, l_q={l_q}, l_r_q={l_r_q}, thr_supp={thr_supp}")
                            except Exception as e:
                                print(f"Error generating dataset at {path} with {bias_type}={value}, sy={sy}, l_q={l_q}, l_r_q={l_r_q}, thr_supp={thr_supp}")
                                print(e)

    # Generate datasets for combinations of two and three biases
    for comb_size in [2, 3]:
        for bias_comb in combinations(bias_types.keys(), comb_size):
            for bias_values in zip(*[bias_types[bias] for bias in bias_comb]):
                for sy in sy_values:
                    for l_q in l_q_values:
                        for l_r_q in l_r_q_values:
                            for thr_supp in thr_supp_values:
                                value_str = "_".join(str(val).replace('.', '_') if not isinstance(val, bool) else ("true" if val else "false") for val in bias_values)
                                path = f"{base_path}{'_'.join(bias_comb)}_{value_str}_sy_{sy}_lq_{l_q}_lrq_{l_r_q}_thrsupp_{thr_supp}"
                                kwargs = {
                                    "path": path,
                                    "dim": dim,
                                    "sy": sy,
                                    "l_q": l_q,
                                    "l_r_q": l_r_q,
                                    "thr_supp": thr_supp
                                }
                                for bias_type, value in zip(bias_comb, bias_values):
                                    kwargs[bias_type] = value
                                try:
                                    biasondemand.generate_dataset(**kwargs)
                                    print(f"Biased dataset generated at {path} with biases {dict(zip(bias_comb, bias_values))}, sy={sy}, l_q={l_q}, l_r_q={l_r_q}, thr_supp={thr_supp}")
                                except Exception as e:
                                    print(f"Error generating dataset at {path} with biases {dict(zip(bias_comb, bias_values))}, sy={sy}, l_q={l_q}, l_r_q={l_r_q}, thr_supp={thr_supp}")
                                    print(e)

def generate_ds(param_dict):
    print("Start creation dataset.", '\n')
    biasondemand.generate_dataset(**param_dict)
def generate_article_datasets():
    all_dicts = {}

    base_path = "datasets/combinations/biased/"
    dim = 10000

    # Define the bias types and their ranges
    bias_types = {
        "l_y": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # Hist bias on Y
        "l_yr": [0.1, 1.5, 3, 6, 9],
        "l_m_y": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # Measurement bias on Y
        "l_m_yr": [0.1, 1.5, 3, 6, 9],  # Measurement bias on Y
        "l_h_r": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # Hist bias on R
        "l_h_rr": [0.1, 1.5, 3, 6, 9],  # Hist bias on R reduced
        "l_h_q": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # Hist bias on Q
        "l_h_qr": [0.1, 1.5, 3, 6, 9],
        "l_m": [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],  # Measurement bias on R
        "l_mr": [0.1, 1.5, 3, 6, 9],  # Measurement bias on R
        "p_u": [0.5, 0.3, 0.1, 0.01, 0.008, 0.006, 0.003],  # Sample bias
        "p_ur": [0.5, 0.1, 0.008, 0.003],  # Sample bias
        "p_u_r": [0.9, 0.7, 0.5, 0.3],  # Sample bias for representation
        "l_r": [True, False],  # Representation bias
        "l_o": [True, False],  # Omission
        "l_y_b": [0]  # Lambda coefficient for interaction proxy bias, i.e., historical bias on the label y with lower values of y for individuals in group A=1 with high values for the feature R
    }


    # S1 - no historical bias
    # no additional bias
    all_dicts['S1A'] = {"path": f"{base_path}S1A", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurment on R
    for v in bias_types["l_m"]:
        ind = bias_types["l_m"].index(v)
        all_dicts[f'S1B{ind}'] = {"path": f"{base_path}S1B{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": v, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # omission
    all_dicts['S1C'] = {"path": f"{base_path}S1C", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                    "l_m": 0, "p_u": 1, "l_r": False, "l_o": True, "l_y_b": 0,
                    "l_q": 2, "sy": 2, "l_r_q": 0}

    # sample
    for v in bias_types["p_u"]:
        ind = bias_types["p_u"].index(v)
        all_dicts[f'S1D{ind}'] = {"path": f"{base_path}S1D{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": v, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_m_y"]:
        ind = bias_types["l_m_y"].index(v)
        all_dicts[f'S1E{ind}'] = {"path": f"{base_path}S1E{ind}", "dim": 10000, "l_y": 0, "l_m_y": v, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # representation bias p_u(variable)
    for v in bias_types["p_u_r"]:
        ind = bias_types["p_u_r"].index(v)
        all_dicts[f'S1F{ind}'] = {"path": f"{base_path}S1F{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": v, "l_r": True, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    

    # S2 historical bias on R
    # no additional bias
    for v in bias_types["l_h_r"]:
        ind = bias_types["l_h_r"].index(v)
        all_dicts[f'S2A{ind}'] = {"path": f"{base_path}S2A{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # measurment on R
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        for v2 in bias_types["l_mr"]:
            ind2 = bias_types["l_mr"].index(v2)
            all_dicts[f'S2B{ind}_{ind2}'] = {"path": f"{base_path}S2B{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": v2, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # omission
    for v in bias_types["l_h_r"]:
        ind = bias_types["l_h_r"].index(v)
        all_dicts[f'S2C{ind}'] = {"path": f"{base_path}S2C{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": True, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # sample
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            all_dicts[f'S2D{ind}_{ind2}'] = {"path": f"{base_path}S2D{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": 0, "p_u": v2, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            all_dicts[f'S2E{ind}_{ind2}'] = {"path": f"{base_path}S2E{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": v2, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # representation bias p_u
    for v in bias_types["l_h_rr"]:
        ind = bias_types["l_h_rr"].index(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            all_dicts[f'S2F{ind}_{ind2}'] = {"path": f"{base_path}S2F{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": v, "l_h_q": 0,
                  "l_m": 0, "p_u": v2, "l_r": True, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # S3 - Historical bias on Y
    # no additional bias
    for v in bias_types["l_y"]:
        ind = bias_types["l_y"].index(v)
        all_dicts[f'S3A{ind}'] = {"path": f"{base_path}S3A{ind}", "dim": 10000, "l_y": v, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # measurment on R
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        for v2 in bias_types["l_mr"]:
            ind2 = bias_types["l_mr"].index(v2)
            all_dicts[f'S3B{ind}_{ind2}'] = {"path": f"{base_path}S3B{ind}_{ind2}", "dim": 10000, "l_y": v, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": v2, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # omission
    for v in bias_types["l_y"]:
        ind = bias_types["l_y"].index(v)
        all_dicts[f'S3C{ind}'] = {"path": f"{base_path}S3C{ind}", "dim": 10000, "l_y": v, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": True, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # sample
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            all_dicts[f'S3D{ind}_{ind2}'] = {"path": f"{base_path}S3D{ind}_{ind2}", "dim": 10000, "l_y": v, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": v2, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            all_dicts[f'S3E{ind}_{ind2}'] = {"path": f"{base_path}S3E{ind}_{ind2}", "dim": 10000, "l_y": v, "l_m_y": v2, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # representation bias p_u
    for v in bias_types["l_yr"]:
        ind = bias_types["l_yr"].index(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            all_dicts[f'S3F{ind}_{ind2}'] = {"path": f"{base_path}S3F{ind}_{ind2}", "dim": 10000, "l_y": v, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": 0,
                  "l_m": 0, "p_u": v2, "l_r": True, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    # S4 - historical bias on Q
    # no additional bias
    for v in bias_types["l_h_q"]:
        ind = bias_types["l_h_q"].index(v)
        all_dicts[f'S4A{ind}'] = {"path": f"{base_path}S4A{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurment on R
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        for v2 in bias_types["l_mr"]:
            ind2 = bias_types["l_mr"].index(v2)
            all_dicts[f'S4B{ind}_{ind2}'] = {"path": f"{base_path}S4B{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": v2, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # omission
    for v in bias_types["l_h_q"]:
        ind = bias_types["l_h_q"].index(v)
        all_dicts[f'S4C{ind}'] = {"path": f"{base_path}S4C{ind}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": True, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # sample
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        for v2 in bias_types["p_ur"]:
            ind2 = bias_types["p_ur"].index(v2)
            all_dicts[f'S4D{ind}_{ind2}'] = {"path": f"{base_path}S4D{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": 0, "p_u": v2, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # measurement bias on Y (P_Y as target). Performance are calculated on Y
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        for v2 in bias_types["l_m_yr"]:
            ind2 = bias_types["l_m_yr"].index(v2)
            all_dicts[f'S4E{ind}_{ind2}'] = {"path": f"{base_path}S4E{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": v2, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": 0, "p_u": 1, "l_r": False, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}

    # representation bias p_u
    for v in bias_types["l_h_qr"]:
        ind = bias_types["l_h_qr"].index(v)
        for v2 in bias_types["p_u_r"]:
            ind2 = bias_types["p_u_r"].index(v2)
            all_dicts[f'S4F{ind}_{ind2}'] = {"path": f"{base_path}S4F{ind}_{ind2}", "dim": 10000, "l_y": 0, "l_m_y": 0, "thr_supp": 1, "l_h_r": 0, "l_h_q": v,
                  "l_m": 0, "p_u": v2, "l_r": True, "l_o": False, "l_y_b": 0,
                  "l_q": 2, "sy": 2, "l_r_q": 0}


    for key, value in all_dicts.items():
        generate_ds(value)


if __name__ == "__main__":
    generate_article_datasets()
