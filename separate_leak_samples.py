import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('multiphase_data.csv')
df = df.fillna(0)

# UPDATED: Removed 'P_leak' to prevent data leakage. 
# The model will learn to deduce leak pressure dynamically via the physical loss!
feature_cols = [
    'M_in', 'M_out', 'P_in', 'P_out', 'Pipe_Length', 
    'Pipe_Diameter', 'P_sur', 'Liquid_Inlet_Frac', 'Liquid_Outlet_Frac', 
    'Mix_Density', 'Mix_Viscosity', 'C_d'
]

# CRITICAL FIX: Include regression targets so SMOTE correctly interpolates them for synthetic leaks
target_cols = ['leak_size', 'leak_location']

# Combine features and regression targets into one matrix for SMOTE
X_combined = df[feature_cols + target_cols].values
y_class = df['leak_class'].values

print("Before SMOTE:")
print(f"Total samples: {len(y_class)}")
print(f"Non-leak (0): {np.sum(y_class == 0)}")
print(f"Leak (1): {np.sum(y_class == 1)}\n")

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_class_resampled = smote.fit_resample(X_combined, y_class)

print("After SMOTE:")
print(f"Total samples: {len(y_class_resampled)}")
print(f"Non-leak (0): {np.sum(y_class_resampled == 0)}")
print(f"Leak (1): {np.sum(y_class_resampled == 1)}\n")

# Create DataFrame with features AND all targets restored
df_resampled = pd.DataFrame(X_resampled, columns=feature_cols + target_cols)
df_resampled['leak_class'] = y_class_resampled

# Separate into leak and non-leak (keeping ALL columns so the model can train)
df_non_leak = df_resampled[df_resampled['leak_class'] == 0]
df_leak = df_resampled[df_resampled['leak_class'] == 1]

# Save to CSV
df_non_leak.to_csv('non_leak_samples.csv', index=False)
df_leak.to_csv('leak_samples.csv', index=False)

# Save the combined balanced dataset (You should point your dataset.py to this file!)
df_resampled.to_csv('multiphase_data_balanced.csv', index=False)

print(f"Saved {len(df_non_leak)} non-leak samples to non_leak_samples.csv")
print(f"Saved {len(df_leak)} leak samples to leak_samples.csv")
print(f"Saved {len(df_resampled)} combined balanced samples to multiphase_data_balanced.csv")