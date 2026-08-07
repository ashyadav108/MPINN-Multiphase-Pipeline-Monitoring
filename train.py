import torch
import torch.optim as optim
from model import MPINN, compute_mpinn_loss
from dataset import prepare_dataloader

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    # Load and preprocess data
    dataloader, scaler, feature_cols = prepare_dataloader('multiphase_data.csv', batch_size=256)
    input_dim = len(feature_cols)

    # Initialize the Physics-Informed Neural Network
    model = MPINN(input_dim=input_dim).to(device)

    # Setup Adam Optimizer and Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    epochs = 1000
    print(f"\nBeginning training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Iterate through batches
        for batch_features, (y_c_true, y_s_true, y_l_true), physics_data in dataloader:
            
            batch_features = batch_features.to(device)
            y_c_true = y_c_true.to(device) 
            y_s_true = y_s_true.to(device)
            y_l_true = y_l_true.to(device)
            
            physics_data = {k: v.to(device) for k, v in physics_data.items()} 
            
            optimizer.zero_grad()
            
            y_c_pred, y_s_pred, y_l_pred = model(batch_features)
            
            loss = compute_mpinn_loss(
                y_c_pred, y_s_pred, y_l_pred,
                y_c_true, y_s_true, y_l_true,
                physics_data,
                alpha=1.0,         
                lambda_phys=0.05,  
                beta1=1e-6,        
                beta2=1e-4,        
                beta3=1e-14        
            )
            
            loss.backward()
            
            # Gradient Clipping: Prevents massive gradient explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Total Composite Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'trained_mpinn.pth')
    print("\nTraining complete! Model weights saved to 'trained_mpinn.pth'")

if __name__ == "__main__":
    train()
