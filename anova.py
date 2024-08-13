import pandas as pd
from scipy.stats import f_oneway

# Example data (replace with your actual results)
results = {
    'Model A': [79.47,79.181,79.456], 
    'Model B': [78.24,78.45,78.26],
    'Model C': [77.04,77.28,77.19],
    'Model D': [76.96,76.85,77.02],
    "Model E": [78.005,78.102,78.343]
}

df = pd.DataFrame(results)
model_a = df['Model A']
model_b = df['Model B']
model_c = df['Model C']
model_d = df['Model D']
model_e = df['Model E']
f_statistic, p_value = f_oneway(model_a, model_b, model_c, model_d, model_e, axis=0)

print("F-statistic:", f_statistic)
print("P-value:", p_value)
