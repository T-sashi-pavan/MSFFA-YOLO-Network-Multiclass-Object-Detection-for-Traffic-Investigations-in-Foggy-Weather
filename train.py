import torch
import torch.optim as optim
from models.msffa_yolo import MSFFA_YOLO
from utils.dataset_loader import create_dataloader
from loss.combined_loss import TotalLoss
import os

def train():
    # settings
    epochs = 10  # More epochs for real data
    batch_size = 4  # Larger batch size
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data - Auto-detect which data to use
    # If train data exists, use it; otherwise use test data
    if os.path.exists("data/foggy_cityscapes/FC01/images/train") and len(os.listdir("data/foggy_cityscapes/FC01/images/train")) > 0:
        img_dir = "data/foggy_cityscapes/FC01/images/train"
        lbl_dir = "data/foggy_cityscapes/FC01/labels/train"
        print("Using TRAINING dataset...")
    else:
        img_dir = "data/foggy_cityscapes/FC01/images/test"
        lbl_dir = "data/foggy_cityscapes/FC01/labels/test"
        print("Using TEST dataset (no training data found)...")
    
    try:
        train_loader = create_dataloader(img_dir, lbl_dir, batch_size=batch_size)
        print(f"Dataloader created with {len(train_loader)} batches")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        raise
    
    # Model
    try:
        model = MSFFA_YOLO(num_classes=8).to(device)
        print(f"Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        raise
    
    # Loss & Optimizer
    try:
        criterion = TotalLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        print(f"Optimizer created")
    except Exception as e:
        print(f"Error creating optimizer: {e}")
        raise

    print("Starting training...")
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for i, (imgs, targets) in enumerate(train_loader):
                try:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    
                    # Forward
                    yolo_preds, restored_img = model(imgs)
                    
                    # Loss Calculation
                    loss = criterion(restored_img, imgs, yolo_preds, targets)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if (i + 1) % 5 == 0 or (i + 1) == len(train_loader):
                        print(f"   Epoch {epoch+1} [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                    
                except Exception as e:
                    print(f"  Error in batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            print(f"\nEpoch [{epoch+1}/{epochs}] complete. Average Loss: {total_loss/len(train_loader):.4f}")
    
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    train()

