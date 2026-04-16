# 🚀 COMPLETE SETUP GUIDE FOR NEW USERS (For Your Friend)

**If you just cloned this repo, follow this guide exactly!**

---

## ❓ FIRST: Do You Have Training Images?

### **Scenario A: NO Training Images (Most Common)**
✅ Use dummy data generator (no file transfer needed)
```bash
python create_dummy_dataset.py
```
**Time:** 2-3 minutes
**Result:** 20 synthetic foggy images ready for training
**No additional files needed!**

---

### **Scenario B: YES You Have Training Images**
Contact the original developer and ask for `data.zip` file
- This is ~47MB 
- Share via: Google Drive, Dropbox, WeTransfer, or similar
- Extract into your cloned repo

---

## 📝 STEP-BY-STEP SETUP

### **Step 1: Clone the Repository**

**Windows:**
```bash
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git
cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather\Foggy_Object_Detection
```

**Mac/Linux:**
```bash
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git
cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather/Foggy_Object_Detection
```

---

### **Step 2: Create Virtual Environment**

**Windows PowerShell/CMD:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux Terminal:**
```bash
python3 -m venv venv
source venv/bin/activate
```

✅ You should see `(venv)` prefix in terminal now

---

### **Step 3: Install All Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- PyTorch + CUDA support
- YOLOv8 detection framework
- OpenCV for image processing
- Streamlit for web interface
- And more...

**⏱️ Time:** 5-10 minutes (depends on internet)

---

### **Step 4: Create Directory Structure**

```bash
python setup_dirs.py
```

**This creates:**
- `data/foggy_cityscapes/FC01/images/train/`
- `data/foggy_cityscapes/FC01/labels/train/`
- `results/` folder
- `weights/` folder

---

### **Step 5: Setup Dataset (Choose ONE)**

#### **Option A: Auto-Generate Fake Data (QUICKEST) ⭐**
```bash
python create_dummy_dataset.py
```
- Creates 20 synthetic images automatically
- Perfect for testing the model works
- Takes 2-3 minutes
- **No file transfer needed!**

#### **Option B: Use Real Dataset**
If you have `data.zip` from the developer:
```bash
# Extract the ZIP file
unzip data.zip
# or on Windows
# Right-click data.zip > Extract All

# Verify it exists
ls data/foggy_cityscapes/FC01/images/train/
```

---

### **Step 6: Quick Test (5 minutes)**

Test that everything works:

```bash
# Test 1: Simple detection on a sample image
python main.py

# Test 2: Test fog removal
python test_restoration.py

# Test 3: Full pipeline (fog removal + detection)
python test_full_pipeline.py
```

**You should see output images:**
- `result.jpg` (YOLOv8 detection)
- `test_comparison.png` (fog removal before/after)
- `test_with_detections.png` (fog removed + objects detected)

✅ If you see these files, everything works!

---

## 🎓 Now Run the Project

### **Option 1: Web Interface (Recommended!!)** 🌐

```bash
streamlit run app_streamlit.py
```

Then open your browser:
```
http://localhost:8501
```

**Features:**
- Upload foggy images
- See detection results with bounding boxes
- Adjust confidence threshold
- Download results

---

### **Option 2: Train the Model**

```bash
python train.py
```

**This:**
- Uses your dataset (real or dummy)
- Trains for ~10 epochs (15-30 minutes on CPU)
- Saves weights to `weights/msffa_yolo_epoch_*.pth`
- Saves logs to `train.log`

To stop training: `Ctrl+C`

---

### **Option 3: Test on Your Image**

```bash
# Place your image in current folder as "my_image.jpg"
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('my_image.jpg')
results[0].save('output.jpg')
print('Output saved to output.jpg')
"
```

---

## 🎯 Recommended Workflow

### **For Testing (Get it Working in 15 minutes):**
1. `python setup_dirs.py`
2. `python create_dummy_dataset.py`
3. `streamlit run app_streamlit.py`
4. Upload a test image in the web interface

### **For Training (Get it Ready for Production):**
1. Get real `data.zip` from developer
2. Extract it
3. `python train.py`
4. Wait for training to complete
5. Use new weights: `python app_streamlit.py`

### **For Production Deployment:**
1. Use pre-trained weights from developer (or your trained weights)
2. Run: `streamlit run app_streamlit.py`
3. Share the URL with others to use

---

## ⚠️ IMPORTANT: About Data and Weights

### **Why aren't images and weights in GitHub?**

GitHub has file size limits. Storing large files slows down cloning.

```
GitHub has:             NOT on GitHub:
✅ All Python code      ❌ Training images (47MB)
✅ Model architectures  ❌ Trained weights (476MB)
✅ Scripts              ❌ Generated results
✅ Configuration
```

### **So What Do You Do?**

**For Images:**
- Option 1: Auto-generate with `python create_dummy_dataset.py` ✅ EASIEST
- Option 2: Ask developer for `data.zip` and extract

**For Weights:**
- Option 1: Let YOLOv8 auto-download (happens automatically)
- Option 2: Ask developer for pre-trained `weights/` folder

---

## 🆘 Troubleshooting

### **Problem: Command not found / ModuleNotFoundError**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Then reinstall requirements
pip install -r requirements.txt
```

### **Problem: "No module named torch"**
```bash
pip install torch torchvision
```

### **Problem: Streamlit won't start**
```bash
pip install streamlit
streamlit run app_streamlit.py --logger.level=warning
```

### **Problem: Out of memory / Slow training**
This is **normal on CPU**. Options:
1. Use GPU (NVIDIA + CUDA installed)
2. Reduce batch size in `train.py`
3. Reduce image size from 640 to 512

### **Problem: "data/foggy_cityscapes/FC01 not found"**
```bash
python setup_dirs.py
python create_dummy_dataset.py
```

---

## ✅ Success Checklist

After following this guide, you should have:

- [x] Virtual environment activated
- [x] All dependencies installed (`torch`, `ultralytics`, etc.)
- [x] Directories created
- [x] Dataset ready (dummy or real)
- [x] Test scripts running without errors
- [x] Web interface accessible at `http://localhost:8501`
- [x] Able to upload images and get detections

**If all checkboxes are checked, you're ready to go!** 🎉

---

## 📊 Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Setup
python setup_dirs.py
python create_dummy_dataset.py

# Quick tests
python main.py
python test_restoration.py
python test_full_pipeline.py

# Web interface
streamlit run app_streamlit.py

# Training
python train.py

# Stop any process
Ctrl + C
```

---

## 🎓 What Each Script Does

| Script | Purpose | Time |
|--------|---------|------|
| `setup_dirs.py` | Create folder structure | <1 min |
| `create_dummy_dataset.py` | Generate synthetic images | 2-3 min |
| `main.py` | Simple YOLOv8 detection | 2-3 min |
| `test_restoration.py` | Test fog removal | 1 min |
| `test_full_pipeline.py` | Test fog removal + detection | 2 min |
| `train.py` | Train the model | 30-60 min (CPU) |
| `app_streamlit.py` | Web interface | Forever (until you stop it) |

---

## 🤝 Need Help?

1. Check this file again - most answers are here!
2. Look at error message - Google it if unclear
3. Check troubleshooting section above
4. Ask the developer for help

---

**Good luck! You've got this!** 🚀

---

*Last Updated: April 2026*
*For: First-time users who cloned the repository*
