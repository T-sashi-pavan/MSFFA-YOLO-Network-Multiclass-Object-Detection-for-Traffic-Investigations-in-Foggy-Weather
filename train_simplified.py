"""
Simplified training script for MSFFA-YOLO
Uses only the restoration loss initially (since we don't have paired clear images)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from models.msffa_yolo import MSFFA_YOLO
from utils.dataset_loader import create_dataloader
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob

class FoggyDataset(Dataset):
    """
    Dataset for foggy images - uses the image itself as reconstruction target
    (Autoencoder style training)
    """
    def __init__(self, img_dir, target_size=512):
        self.img_dir = img_dir
        self.target_size = target_size
        self.images = glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True) + \
                      glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True)
        print(f"Found {len(self.images)} images in {img_dir}")
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Return black image if failed
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            # Resize to target size
            img = cv2.resize(img, (self.target_size, self.target_size))
        
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor

def train():
    print("=" * 70)
    print("TRAINING MSFFA-YOLO MODEL")
    print("=" * 70)
    
    # Settings
    epochs = 10
    batch_size = 2  # Reduced for CPU
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Data path
    if os.path.exists("data/foggy_cityscapes/FC01/images/train"):
        img_dir = "data/foggy_cityscapes/FC01/images/train"
        print(f"Using training dataset: {img_dir}")
    else:
        img_dir = "Foggy_Driving/leftImg8bit"
        print(f"Using Foggy Driving dataset: {img_dir}")
    
    # Create dataset and loader
    try:
        dataset = FoggyDataset(img_dir, target_size=512)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"Created dataloader with {len(train_loader)} batches")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    # Model
    try:
        model = MSFFA_YOLO(num_classes=8).to(device)
        print(f"Model created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Loss - Simple reconstruction loss for now
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Optimizer: Adam (lr={lr})")
    print("\nStarting training...\n")
    
    # Training loop
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, imgs in enumerate(train_loader):
                try:
                    imgs = imgs.to(device)
                    
                    # Forward pass
                    yolo_preds, restored_img = model(imgs)
                    
                    # Simple reconstruction loss: restored should match input
                    # (In a real supervised setup, we'd use clear image pairs)
                    loss = criterion(restored_img, imgs)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
                    
                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch+1} completed - Average Loss: {avg_loss:.6f}\n")
            
            # Save checkpoint every epoch
            checkpoint_path = f"weights/msffa_yolo_epoch_{epoch+1}.pth"
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}\n")
        
        # Save final model
        final_path = "weights/msffa_yolo_final.pth"
        torch.save(model.state_dict(), final_path)
        print("=" * 70)
        print(f"✓ TRAINING COMPLETE!")
        print(f"✓ Final model saved to: {final_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
