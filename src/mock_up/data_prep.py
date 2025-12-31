import numpy as np
import pandas as pd

def generate_hierarchical_data():
    np.random.seed(42)
    n = 1000
    t = np.arange(n)
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    # Now: 1 H1, 1 H2, 1 H3
    h1_groups = ['h1g1']
    h2_groups = ['h2g1']
    h3_groups = ['h3g1']

    data_dict = {}

    # Hierarchical effects
    h1_effects = {'h1g1': 50}
    h2_effects = {'h2g1': 5}
    h3_effects = {'h3g1': 2}

    for h1 in h1_groups:
        for h2 in h2_groups:
            for h3 in h3_groups:
                group_name = f"{h1}_{h2}_{h3}"

                baseline = h1_effects[h1] + h2_effects[h2] + h3_effects[h3]
                trend = np.random.normal(0.03, 0.01) * t
                seasonality = np.random.normal(8, 2) * np.sin(2 * np.pi * t / 52)
                noise = np.random.normal(0, 1.5, n)

                ts = baseline + trend + seasonality + noise
                data_dict[group_name] = ts

    df = pd.DataFrame(data_dict, index=dates)

    hierarchy_map = []
    for col in df.columns:
        h1, h2, h3 = col.split('_')
        hierarchy_map.append({'group': col, 'h1': h1, 'h2': h2, 'h3': h3})

    return df, pd.DataFrame(hierarchy_map)
