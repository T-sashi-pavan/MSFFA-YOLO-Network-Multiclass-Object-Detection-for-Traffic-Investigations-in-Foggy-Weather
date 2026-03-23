"""
IMPROVED TRAINING: Use histogram equalization as restoration target
Instead of requiring paired clear images, we enhance the foggy images themselves
"""
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os
import glob
from models.msffa_yolo import MSFFA_YOLO
from torch.utils.data import Dataset, DataLoader

class EnhancedFoggyDataset(Dataset):
    """
    Dataset where:
    - Input: Original foggy image
    - Target: Histogram-equalized/enhanced version of same image
    This forces the model to learn enhancement, not just output white
    """
    def __init__(self, img_dir, target_size=512):
        self.img_dir = img_dir
        self.target_size = target_size
        self.images = glob.glob(os.path.join(img_dir, "**/*.png"), recursive=True) + \
                      glob.glob(os.path.join(img_dir, "**/*.jpg"), recursive=True)
        print(f"Found {len(self.images)} images")
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size))
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Create enhancement target: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray).astype(np.float32) / 255.0
        
        # Convert enhanced grayscale to BGR-like enhancement
        enhanced_bgr = cv2.cvtColor((enhanced_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # More sophisticated: Blend original with enhanced to preserve colors while enhancing contrast
        enhanced_rgb = 0.7 * img_rgb + 0.3 * enhanced_rgb
        enhanced_rgb = np.clip(enhanced_rgb, 0, 1)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        target_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1)
        
        return img_tensor, target_tensor

def train_improved():
    print("=" * 80)
    print("IMPROVED MSFFA-YOLO TRAINING (With Histogram Enhancement)")
    print("=" * 80)
    
    epochs = 15  # More epochs since we have a clearer training target
    batch_size = 2
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Data
    if os.path.exists("data/foggy_cityscapes/FC01/images/train"):
        img_dir = "data/foggy_cityscapes/FC01/images/train"
    else:
        img_dir = "Foggy_Driving/leftImg8bit"
    
    print(f"Dataset: {img_dir}")
    
    try:
        dataset = EnhancedFoggyDataset(img_dir, target_size=512)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        print(f"Created dataloader with {len(train_loader)} batches\n")
    except Exception as e:
        print(f"Error creating dataloader: {e}")
        return
    
    # Model
    model = MSFFA_YOLO(num_classes=8).to(device)
    
    # Try to load previous weights as initialization
    if os.path.exists("weights/msffa_yolo_final.pth"):
        try:
            model.load_state_dict(torch.load("weights/msffa_yolo_final.pth", map_location=device))
            print("✓ Initialized from previous training\n")
        except:
            print("Could not load previous weights, starting fresh\n")
    
    criterion = nn.L1Loss()  # L1 loss is better for image restoration than MSE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting improved training...\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (imgs, targets) in enumerate(train_loader):
            try:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                # Forward
                yolo_preds, restored_img = model(imgs)
                
                # Loss: Restored should match the enhanced target
                # Scale outputs to [0,1] for comparison with target
                restored_scaled = (restored_img + 1) / 2.0  # Tanh [-1,1] → [0,1]
                restored_scaled = torch.clamp(restored_scaled, 0, 1)
                
                loss = criterion(restored_scaled, targets)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n→ Epoch {epoch+1} Average Loss: {avg_loss:.6f}\n")
        
        # Save checkpoint
        os.makedirs("weights", exist_ok=True)
        checkpoint_path = f"weights/msffa_yolo_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✓ Saved: {checkpoint_path}\n")
    
    # Save final
    final_path = "weights/msffa_yolo_final.pth"
    torch.save(model.state_dict(), final_path)
    
    print("=" * 80)
    print("✓ IMPROVED TRAINING COMPLETE!")
    print(f"✓ Final model: {final_path}")
    print("=" * 80)
    print("\nThe model now learns to enhance foggy images instead of outputting white!")
    print("Restart Streamlit to see the improvements: http://localhost:8501")

if __name__ == "__main__":
    train_improved()
