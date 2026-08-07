import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import r2_score, mean_absolute_error
from model import MPINN
from dataset import prepare_dataloader

# Configure publication-grade plot styles
plt.rcParams.update({                   
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,          # Added subtle gridlines for readability
    'grid.alpha': 0.3
})
fig_dpi = 300  

def run_paper_workflow():  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STEP 1: Loading Dataset and Trained MPINN on {device} ---")
    
    dataloader, scaler, feature_cols = prepare_dataloader('multiphase_data.csv', batch_size=50)
    test_features, (y_c_true, y_s_true, y_l_true), _ = next(iter(dataloader))
    
    test_features = test_features.to(device)
    
    input_dim = len(feature_cols)
    model = MPINN(input_dim=input_dim).to(device)
    
    model.load_state_dict(torch.load('trained_mpinn.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        y_c_pred, y_s_pred, y_l_pred = model(test_features)
        
    s_true = y_s_true.cpu().numpy().flatten()
    s_pred = y_s_pred.cpu().numpy().flatten()
    l_true = y_l_true.cpu().numpy().flatten()
    l_pred = y_l_pred.cpu().numpy().flatten()

    print("\n--- STEP 2: Generating Parity & Performance Plots ---")
    
    # 1. Leak Location Parity Plot
    plt.figure(figsize=(6, 6))  
    plt.scatter(l_true, l_pred, color='#1f77b4', alpha=0.7, edgecolors='k', s=45, label='MPINN Predictions')  
    
    # IMPROVEMENT: Calculate metrics to display
    loc_r2 = r2_score(l_true, l_pred)
    loc_mae = mean_absolute_error(l_true, l_pred)
    
    # IMPROVEMENT: Ensure square aspect ratio and perfect y=x line
    min_val, max_val = min(l_true.min(), l_pred.min()), max(l_true.max(), l_pred.max())
    # Add a little padding to the limits
    padding = (max_val - min_val) * 0.05
    plt.xlim(min_val - padding, max_val + padding)
    plt.ylim(min_val - padding, max_val + padding)
    plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'k--', lw=2, label='Ideal Fit (y=x)')
    
    plt.gca().set_aspect('equal', adjustable='box') # Forces the 45-degree angle
    
    plt.title('Leak Location Estimation Parity')
    plt.xlabel('Ground Truth Location (m)')
    plt.ylabel('Predicted Location (m)')
    plt.legend(loc='upper left')
    
    # IMPROVEMENT: Add metric text box
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    plt.text(0.95, 0.05, f"R² = {loc_r2:.2f}\nMAE = {loc_mae:.1f} m", 
             transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)
    
    plt.tight_layout() 
    plt.savefig('fig_location_parity.png', dpi=fig_dpi)
    plt.close()
    
    # 2. Leak Size Parity Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(s_true, s_pred, color='#ff7f0e', alpha=0.7, edgecolors='k', s=45, label='MPINN Predictions')
    
    size_r2 = r2_score(s_true, s_pred)
    size_mae = mean_absolute_error(s_true, s_pred)
    
    min_s, max_s = min(s_true.min(), s_pred.min()), max(s_true.max(), s_pred.max())
    padding_s = (max_s - min_s) * 0.05
    plt.xlim(min_s - padding_s, max_s + padding_s)
    plt.ylim(min_s - padding_s, max_s + padding_s)
    plt.plot([min_s - padding_s, max_s + padding_s], [min_s - padding_s, max_s + padding_s], 'k--', lw=2, label='Ideal Fit (y=x)')
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title('Leak Dimension Estimation Parity')
    plt.xlabel('Ground Truth Size (m)')
    plt.ylabel('Predicted Size (m)')
    plt.legend(loc='upper left')
    
    plt.text(0.95, 0.05, f"R² = {size_r2:.4f}\nMAE = {size_mae:.4f} m", 
             transform=plt.gca().transAxes, fontsize=11, 
             verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)
             
    plt.tight_layout()
    plt.savefig('fig_size_parity.png', dpi=fig_dpi)
    plt.close() 
    print("Saved: 'fig_location_parity.png' & 'fig_size_parity.png'")

    print("\n--- STEP 3: Executing SHAP Explainable AI Framework ---")
    class SHAPWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            c, _, _ = self.m(x)
            return c

    wrapped_model = SHAPWrapper(model).to(device)
    wrapped_model.eval()
    
    # IMPROVEMENT: Use a small background dataset to prevent memory overload
    background = test_features[:10]
    test_samples = test_features[10:]
    
    explainer = shap.DeepExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(test_samples) 
    
    shap_vals_plot = shap_values[0] if isinstance(shap_values, list) else shap_values 
    if len(shap_vals_plot.shape) == 3:
        shap_vals_plot = shap_vals_plot.squeeze(axis=-1) 

    plt.figure(figsize=(9, 6)) 
    shap.summary_plot(shap_vals_plot, test_samples.cpu().numpy(), feature_names=feature_cols, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance Analysis', pad=15) 
    plt.tight_layout()
    plt.savefig('fig_shap_importance.png', dpi=fig_dpi)
    plt.close()
    print("Saved: 'fig_shap_importance.png'")
    
    print("\nWorkflow Execution Complete. All validation artifacts ready for manuscript integration.")

if __name__ == "__main__":
    run_paper_workflow()