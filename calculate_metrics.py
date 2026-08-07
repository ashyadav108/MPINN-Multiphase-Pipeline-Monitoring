import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, mean_absolute_error, r2_score
from model import MPINN
from dataset import prepare_dataloader

def evaluate_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation metrics on device: {device}")

    dataloader, scaler, feature_cols = prepare_dataloader('multiphase_data.csv', batch_size=200)
    
    all_c_true, all_c_pred = [], [] 
    all_s_true, all_s_pred = [], []
    all_l_true, all_l_pred = [], []
    
    input_dim = len(feature_cols)
    model = MPINN(input_dim=input_dim).to(device)
    
    model.load_state_dict(torch.load('trained_mpinn.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        for batch_features, (y_c_true, y_s_true, y_l_true), _ in dataloader:
            batch_features = batch_features.to(device)
            y_c_pred, y_s_pred, y_l_pred = model(batch_features)
            
            c_pred_binary = (y_c_pred >= 0.5).int()
            
            all_c_true.extend(y_c_true.cpu().numpy().flatten()) 
            all_c_pred.extend(c_pred_binary.cpu().numpy().flatten())
            
            all_s_true.extend(y_s_true.cpu().numpy().flatten())
            all_s_pred.extend(y_s_pred.cpu().numpy().flatten())
            
            all_l_true.extend(y_l_true.cpu().numpy().flatten())
            all_l_pred.extend(y_l_pred.cpu().numpy().flatten())

    all_c_true = np.array(all_c_true)
    all_c_pred = np.array(all_c_pred)
    all_s_true = np.array(all_s_true)
    all_s_pred = np.array(all_s_pred)
    all_l_true = np.array(all_l_true)
    all_l_pred = np.array(all_l_pred)

    acc = accuracy_score(all_c_true, all_c_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(all_c_true, all_c_pred, average='binary', zero_division=0)

    # Filter metrics to evaluate only on active leaks
    leak_mask = (all_c_true == 1)
    
    if np.sum(leak_mask) > 0:
        s_true_leaks = all_s_true[leak_mask]
        s_pred_leaks = all_s_pred[leak_mask]
        l_true_leaks = all_l_true[leak_mask]
        l_pred_leaks = all_l_pred[leak_mask]

        s_rmse = np.sqrt(mean_squared_error(s_true_leaks, s_pred_leaks))
        s_mae = mean_absolute_error(s_true_leaks, s_pred_leaks)
        s_r2 = r2_score(s_true_leaks, s_pred_leaks)

        l_rmse = np.sqrt(mean_squared_error(l_true_leaks, l_pred_leaks))
        l_mae = mean_absolute_error(l_true_leaks, l_pred_leaks)
        l_r2 = r2_score(l_true_leaks, l_pred_leaks)
    else:
        s_rmse, s_mae, s_r2, l_rmse, l_mae, l_r2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    print("\n" + "="*50)
    print("QUANTITATIVE PERFORMANCE EVALUATION TABLE")
    print("="*50)
    print(f"{'Task / Metric':<25} | {'Value':<15}") 
    print("-" * 43)
    print(f"{'Classification Accuracy':<25} | {acc*100:.2f}%")
    print(f"{'Precision':<25} | {precision:.4f}")
    print(f"{'Recall':<25} | {recall:.4f}")
    print(f"{'F1-Score':<25} | {f1:.4f}")
    print("-" * 43)
    print(f"{'Leak Size RMSE':<25} | {s_rmse:.4f}")
    print(f"{'Leak Size MAE':<25} | {s_mae:.4f}")
    print(f"{'Leak Size R²':<25} | {s_r2:.4f}")
    print("-" * 43)
    print(f"{'Leak Location RMSE (m)':<25} | {l_rmse:.2f}")
    print(f"{'Leak Location MAE (m)':<25} | {l_mae:.2f}")
    print(f"{'Leak Location R²':<25} | {l_r2:.2f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_metrics()