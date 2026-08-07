import torch
import numpy as np
import shap
from model import MPINN
from dataset import prepare_dataloader

def get_mc_dropout_uncertainty(model, x, num_samples=50):
    """ Runs Monte Carlo Dropout to estimate prediction uncertainty (95% CI) """
    model.train() # Force dropout active to sample different network states
    predictions_c, predictions_s, predictions_l = [], [], []
    
    with torch.no_grad():
        for _ in range(num_samples):
            y_c, y_s, y_l = model(x)
            predictions_c.append(y_c.cpu().numpy())
            predictions_s.append(y_s.cpu().numpy())
            predictions_l.append(y_l.cpu().numpy())
            
    # Calculate mean and 95% Confidence Interval (1.96 * standard deviation)
    mean_c = np.mean(predictions_c, axis=0)
    ci_c = 1.96 * np.std(predictions_c, axis=0)
    
    mean_s = np.mean(predictions_s, axis=0)
    ci_s = 1.96 * np.std(predictions_s, axis=0)
    
    mean_l = np.mean(predictions_l, axis=0)
    ci_l = 1.96 * np.std(predictions_l, axis=0)
    
    return (mean_c, ci_c), (mean_s, ci_s), (mean_l, ci_l)


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # Load data
    dataloader, scaler, feature_cols = prepare_dataloader('multiphase_data.csv', batch_size=50)
    
    # Grab a batch of data
    batch_features, (batch_y_c, batch_y_s, batch_y_l), _ = next(iter(dataloader))
    
    # Find the first index where a leak actually exists (Class 1)
    leak_indices = (batch_y_c == 1).nonzero(as_tuple=True)[0]
    
    if len(leak_indices) == 0:
        print("No leaks found in this batch. Try running again!")
        return
        
    leak_idx = leak_indices[0].item()
    
    # Select that specific active leak sample
    test_feature = batch_features[leak_idx:leak_idx+1].to(device)
    y_c_true = batch_y_c[leak_idx:leak_idx+1]
    y_s_true = batch_y_s[leak_idx:leak_idx+1]
    y_l_true = batch_y_l[leak_idx:leak_idx+1]
    
    input_dim = len(feature_cols)
    
    # Load Model
    model = MPINN(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load('trained_mpinn.pth', map_location=device))
    
    # Standard Prediction (No Dropout)
    model.eval() 
    with torch.no_grad():
        y_c_pred, y_s_pred, y_l_pred = model(test_feature)

    print("\n" + "="*45)  
    print("MODEL EVALUATION RESULTS (Active Leak Sample)")
    print("="*45)
    
    true_class = int(y_c_true[0].item())
    pred_class = int(y_c_pred[0].item() >= 0.5)
    print(f"True Leak Class:    {true_class} | Predicted Class:    {pred_class} (Prob: {y_c_pred[0].item():.4f})")
    print(f"True Leak Size:     {y_s_true[0].item():.4f} | Predicted Size:     {y_s_pred[0].item():.4f}")
    print(f"True Leak Location: {y_l_true[0].item():.2f} | Predicted Location: {y_l_pred[0].item():.2f}")

    print("\n--- UNCERTAINTY ESTIMATION (95% CI) ---")
    (mean_c, ci_c), (mean_s, ci_s), (mean_l, ci_l) = get_mc_dropout_uncertainty(model, test_feature, num_samples=50)
    print(f"Leak Probability: {mean_c[0][0]:.4f} ± {ci_c[0][0]:.4f}")
    print(f"Leak Size:        {mean_s[0][0]:.4f} ± {ci_s[0][0]:.4f}")
    print(f"Leak Location:    {mean_l[0][0]:.2f} ± {ci_l[0][0]:.2f}")

    print("\n--- CALCULATING SHAP FEATURE IMPORTANCE ---")
    class SHAPWrapper(torch.nn.Module):
        def __init__(self, m): 
            super().__init__() 
            self.m = m 
        def forward(self, x):  
            c, _, _ = self.m(x)
            return c

    wrapped_model = SHAPWrapper(model).to(device)
    wrapped_model.eval() 
    
    # Use a random subset of the batch as background for SHAP
    background = batch_features[:10].to(device)
    explainer = shap.DeepExplainer(wrapped_model, background) 
    shap_values = explainer.shap_values(test_feature) 
    
    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    mean_abs_shap = np.array(mean_abs_shap).flatten()
    
    feature_importance = sorted(zip(feature_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)
    
    print("\nTop 3 Most Important Features for detecting a leak:")
    for rank, (feature, importance) in enumerate(feature_importance[:3], 1):
        print(f"{rank}. {feature} (Impact Score: {float(importance):.4f})")

if __name__ == "__main__":
    evaluate_model()