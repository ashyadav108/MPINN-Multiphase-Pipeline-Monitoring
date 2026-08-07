import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

class MultiphasePipelineDataset(Dataset):
    def __init__(self, features, labels_c, labels_s, labels_l, physics_data):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels_c = torch.tensor(labels_c, dtype=torch.float32).unsqueeze(1)  
        self.labels_s = torch.tensor(labels_s, dtype=torch.float32).unsqueeze(1)
        self.labels_l = torch.tensor(labels_l, dtype=torch.float32).unsqueeze(1)
        
        self.physics_data = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(1) 
                             for k, v in physics_data.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        phys_item = {k: v[idx] for k, v in self.physics_data.items()} 
        return (self.features[idx], 
                (self.labels_c[idx], self.labels_s[idx], self.labels_l[idx]), 
                phys_item)

def prepare_dataloader(csv_file='multiphase_data.csv', batch_size=256):
    print("Loading Data...")
    df = pd.read_csv(csv_file)
    df = df.fillna(0)
    
    # Neural network will only see measurable inputs
    feature_cols = ['M_in', 'M_out', 'P_in', 'P_out', 'Pipe_Length', 
                    'Pipe_Diameter', 'P_sur', 'Liquid_Inlet_Frac', 'Liquid_Outlet_Frac', 
                    'Mix_Density', 'Mix_Viscosity', 'C_d']
    
    X = df[feature_cols].values 
    y_class = df['leak_class'].values 
    y_size = df['leak_size'].values
    y_loc = df['leak_location'].values

    print("Applying SMOTE (Handling Class Imbalance)...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)  
    X_resampled, y_class_resampled = smote.fit_resample(X, y_class)
    
    y_size_resampled = np.where(y_class_resampled == 0, 0.0, 
                                np.interp(np.arange(len(y_class_resampled)), 
                                          np.arange(len(y_size)), y_size))
    y_loc_resampled = np.where(y_class_resampled == 0, 0.0, 
                               np.interp(np.arange(len(y_class_resampled)), 
                                         np.arange(len(y_loc)), y_loc))

    print("Applying StandardScaler (Normalization)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    # UPDATED: Added P_sur, Mix_Density, and C_d based on their array indices
    physics_data = {
        'M_in': X_resampled[:, 0],   
        'M_out': X_resampled[:, 1],     
        'P_in': X_resampled[:, 2],      
        'P_out': X_resampled[:, 3],     
        'L': X_resampled[:, 4],
        'P_sur': X_resampled[:, 6],         # Added for physical leak mass calculation
        'Mix_Density': X_resampled[:, 9],   # Added for physical leak mass calculation
        'C_d': X_resampled[:, 11],          # Added for physical leak mass calculation
        'M_leak_hat': y_size_resampled 
    }
    
    dataset = MultiphasePipelineDataset(X_scaled, y_class_resampled, y_size_resampled, y_loc_resampled, physics_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    
    print(f"Preprocessing Complete! Total balanced samples: {len(X_resampled)}")
    return dataloader, scaler, feature_cols

if __name__ == "__main__":
    # Test the preprocessing script
    dl, scaler, cols = prepare_dataloader()
    features, labels, phys = next(iter(dl))
    print(f"Batch loaded successfully with feature shape: {features.shape}")