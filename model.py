import torch
import torch.nn as nn
import numpy as np

# ==========================================
# Shared Backbone & Task Heads
# ==========================================
class MPINN(nn.Module):
    def __init__(self, input_dim):
        super(MPINN, self).__init__() 
        
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(), 
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Dropout(p=0.1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(64, 1), 
            nn.Sigmoid() 
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts Leak Diameter (meters)
        )
        
        self.location_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Predicts Leak Location percentage
        )

    def forward(self, x): 
        features = self.shared_backbone(x)
        y_c = self.classification_head(features)
        
        # Enforcing physical constraints so size can never be negative
        y_s = torch.relu(self.size_head(features))
        
        # Target Normalization: predict percentage (0.0 to 1.0), scale to max length
        y_l_normalized = torch.sigmoid(self.location_head(features))
        y_l = y_l_normalized * 6500.0 
        
        return y_c, y_s, y_l

# ==========================================
# Dynamic Physics-Informed Loss Function
# ==========================================
def compute_mpinn_loss(y_c_pred, y_s_pred, y_l_pred, 
                       y_c_true, y_s_true, y_l_true, 
                       physics_inputs, 
                       alpha=1.0, lambda_phys=0.1, 
                       beta1=1.0, beta2=1.0, beta3=1.0):
    
    # 1. Data-Driven Losses
    L_cls = nn.BCELoss()(y_c_pred, y_c_true) 
    
    # Restored to L1Loss on raw values
    L_size = nn.L1Loss()(y_s_pred, y_s_true) 
    L_loc = nn.L1Loss()(y_l_pred, y_l_true)
    
    # 2. Extract physics variables
    M_in = physics_inputs['M_in']
    M_out = physics_inputs['M_out']
    P_in = physics_inputs['P_in']
    P_out = physics_inputs['P_out']
    L_pipe = physics_inputs['L']
    
    P_sur = physics_inputs.get('P_sur', 101325.0) 
    Mix_Density = physics_inputs.get('Mix_Density', 800.0)
    C_d = physics_inputs.get('C_d', 0.6)
    
    # 3. DYNAMIC PHYSICS CALCULATION
    P_leak_pred = P_in - (y_l_pred / (L_pipe + 1e-8)) * (P_in - P_out)
    A_leak = np.pi * (y_s_pred / 2.0) ** 2
    delta_p = torch.relu(P_leak_pred - P_sur) 
    M_leak_physics = C_d * A_leak * torch.sqrt(2.0 * Mix_Density * delta_p + 1e-8)
    
    # 4. PHYSICS LOSS TERMS
    actual_lost_mass = torch.relu(M_in - M_out)
    loss_mass = torch.mean((actual_lost_mass - M_leak_physics) ** 2)
    loss_press = torch.mean(torch.relu(P_out - P_in))
    loss_bound = torch.mean(torch.relu(y_l_pred - L_pipe) + torch.relu(-y_l_pred)) 
    
    L_phys = (beta1 * loss_mass) + (beta2 * loss_press) + (beta3 * loss_bound)
    
    # RESTORED: Heavy weight so the neural network respects Leak Size against Physics!
    weight_size = 5000.0 
    
    L_total = L_cls + (weight_size * L_size) + (alpha * L_loc) + (lambda_phys * L_phys)
    
    return L_total