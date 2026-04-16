"""
Quick training script with dataset generation
Trains MSFFA-YOLO model on dummy foggy dataset
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import glob
import time
from pathlib import Path

# Import model
from models.msffa_yolo import MSFFA_YOLO

def create_dummy_dataset(base_path="data", num_train=15, num_val=5):
    """Create dummy foggy dataset with labels"""
    print("\n" + "=" * 70)
    print("GENERATING DUMMY FOGGY DATASET")
    print("=" * 70)
    
    sets = {'train': num_train, 'val': num_val}
    
    for set_type, num_images in sets.items():
        img_dir = os.path.join(base_path, 'foggy_cityscapes', 'FC01', 'images', set_type)
        lbl_dir = os.path.join(base_path, 'foggy_cityscapes', 'FC01', 'labels', set_type)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        
        print(f"\nGenerating {num_images} {set_type} images...")
        
        for i in range(num_images):
            # Create base image (foggy effect)
            img = np.ones((512, 512, 3), dtype=np.uint8) * 180  # Grayish foggy background
            
            # Add some foggy texture
            noise = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.9, noise, 0.1, 0)
            
            # Draw random objects (simulating traffic scene)
            label_str = ""
            num_objects = np.random.randint(2, 6)
            
            for _ in range(num_objects):
                cls_id = np.random.randint(0, 8)  # 8 classes
                x1 = np.random.randint(50, 350)
                y1 = np.random.randint(50, 350)
                w = np.random.randint(40, 150)
                h = np.random.randint(40, 150)
                
                color = tuple(np.random.randint(50, 200, 3).tolist())
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 255, 255), 2)
                
                # YOLO Format: class x_center y_center width height (normalized 0-1)
                xc = (x1 + w/2) / 512
                yc = (y1 + h/2) / 512
                wn = w / 512
                hn = h / 512
                label_str += f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
            
            # Add fog overlay
            fog = np.ones_like(img) * 220
            img = cv2.addWeighted(img, 0.7, fog, 0.3, 0)
            
            # Save image and label
            filename = f"image_{i:04d}"
            img_path = os.path.join(img_dir, filename + ".png")
            lbl_path = os.path.join(lbl_dir, filename + ".txt")
            
            cv2.imwrite(img_path, img)
            with open(lbl_path, 'w') as f:
                f.write(label_str)
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{num_images} images")
    
    print("✓ Dataset generation complete!")
    return True

class FoggyDataset(Dataset):
    """Dataset for foggy images"""
    def __init__(self, img_dir, target_size=512):
        self.img_dir = img_dir
        self.target_size = target_size
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.png"))) + \
                      sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {img_dir}")
        
        print(f"  Dataset size: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.target_size, self.target_size))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Convert BGR to RGB and to tensor
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
        
        return img_tensor

def train_model():
    """Train MSFFA-YOLO model"""
    print("\n" + "=" * 70)
    print("TRAINING MSFFA-YOLO MODEL")
    print("=" * 70)
    
    # Configuration
    epochs = 5  # Quick training
    batch_size = 2
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "weights/msffa_yolo_trained.pth"
    
    print(f"\n📊 Training Configuration:")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    
    # Create datasets
    train_img_dir = "data/foggy_cityscapes/FC01/images/train"
    val_img_dir = "data/foggy_cityscapes/FC01/images/val"
    
    print(f"\n📂 Loading training dataset...")
    train_dataset = FoggyDataset(train_img_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"📂 Loading validation dataset...")
    val_dataset = FoggyDataset(val_img_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"\n🏗️  Building MSFFA-YOLO model...")
    model = MSFFA_YOLO(num_classes=8)
    model = model.to(device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()  # For restoration loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        start_time = time.time()
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            detections, restored = model(images)
            
            # Loss: reconstruction loss (MSE between restored and original)
            loss = criterion(restored, images)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if (batch_idx + 1) % max(1, len(train_loader) // 2) == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                detections, restored = model(images)
                loss = criterion(restored, images)
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        epoch_time = time.time() - start_time
        
        print(f"✓ Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")
        
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  💾 Best model saved to {model_save_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Model saved to: {model_save_path}")
    print("=" * 70)

if __name__ == "__main__":
    # Step 1: Generate dataset
    create_dummy_dataset(num_train=15, num_val=5)
    
    # Step 2: Train model
    train_model()
