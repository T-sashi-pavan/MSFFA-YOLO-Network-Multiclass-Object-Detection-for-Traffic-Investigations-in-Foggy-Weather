# ⚡ QUICK COMMAND REFERENCE

## All Commands Your Friend Needs (Copy & Paste)

---

## 🖥️ WINDOWS Users

```bash
# ===== STEP 1: CLONE =====
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git
cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather\Foggy_Object_Detection

# ===== STEP 2: SETUP ENVIRONMENT =====
python -m venv venv
venv\Scripts\activate

# ===== STEP 3: INSTALL DEPENDENCIES =====
pip install --upgrade pip
pip install -r requirements.txt

# ===== STEP 4: SETUP DIRECTORIES =====
python setup_dirs.py

# ===== STEP 5: GENERATE DATA (NO FILE TRANSFER!) =====
python create_dummy_dataset.py

# ===== STEP 6A: RUN WEB INTERFACE (BEST!) =====
streamlit run app_streamlit.py

# ===== STEP 6B: QUICK TEST (ALTERNATIVE) =====
python main.py
python test_restoration.py
python test_full_pipeline.py

# ===== STEP 7: TRAIN MODEL (OPTIONAL) =====
python train.py

```

---

## 🖥️ MAC/LINUX Users

```bash
# ===== STEP 1: CLONE =====
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git
cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather/Foggy_Object_Detection

# ===== STEP 2: SETUP ENVIRONMENT =====
python3 -m venv venv
source venv/bin/activate

# ===== STEP 3: INSTALL DEPENDENCIES =====
pip install --upgrade pip
pip install -r requirements.txt

# ===== STEP 4: SETUP DIRECTORIES =====
python create_dummy_dataset.py

# ===== STEP 5: GENERATE DATA (NO FILE TRANSFER!) =====
python create_dummy_dataset.py

# ===== STEP 6A: RUN WEB INTERFACE (BEST!) =====
streamlit run app_streamlit.py

# ===== STEP 6B: QUICK TEST (ALTERNATIVE) =====
python main.py
python test_restoration.py
python test_full_pipeline.py

# ===== STEP 7: TRAIN MODEL (OPTIONAL) =====
python train.py

```

---

## 📋 WHAT HAPPENS AT EACH STEP

| Step | Command | What It Does | Time |
|------|---------|-------------|------|
| 1 | `git clone` | Downloads all code | 2 min |
| 2 | `python -m venv venv` | Creates isolated Python | 1 min |
| 3 | `pip install -r requirements.txt` | Installs torch, streamlit, opencv, etc | 5-10 min |
| 4 | `python setup_dirs.py` | Creates folder structure | <1 min |
| 5 | `python create_dummy_dataset.py` | Generates 20 test images | 2-3 min |
| 6A | `streamlit run app_streamlit.py` | Web interface at localhost:8501 | Forever (until stopped) |
| 6B | `python test_*.py` | Quick tests | 1-2 min each |
| 7 | `python train.py` | Train the model | 30-60 min (CPU) |

---

## 🎯 FASTEST PATH (15 MINUTES)

```bash
# Windows
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git && cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather\Foggy_Object_Detection && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python create_dummy_dataset.py && streamlit run app_streamlit.py

# Mac/Linux
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git && cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather/Foggy_Object_Detection && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python create_dummy_dataset.py && streamlit run app_streamlit.py
```

---

## 🧪 TESTING COMMANDS

```bash
# Test 1: Simple detection
python main.py

# Test 2: Fog removal only
python test_restoration.py

# Test 3: Full pipeline (fog removal + detection)
python test_full_pipeline.py

# Test 4: On real dataset (if you have it)
python test_on_real_data.py
```

---

## 🌐 WEB INTERFACE

```bash
# Start the web app
streamlit run app_streamlit.py

# Browser automatically opens at:
# http://localhost:8501

# Features:
# - Upload images
# - Real-time detection
# - Adjust confidence threshold
# - Download results

# To stop: Ctrl+C in terminal
```

---

## 🚂 TRAINING

```bash
# Train the model (30-60 min on CPU, 5-10 min on GPU)
python train.py

# Check training progress in terminal
# Outputs go to: weights/msffa_yolo_epoch_*.pth

# To stop training: Ctrl+C
```

---

## 📊 IF SOMETHING FAILS

```bash
# Check if venv is activated
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# Clear cache and restart
streamlit cache clear
streamlit run app_streamlit.py --logger.level=warning

# Check Python version (need 3.8+)
python --version

# Check if torch installed
python -c "import torch; print(torch.__version__)"

# Update pip
pip install --upgrade pip
```

---

## 🗂️ DIRECTORY STRUCTURE (What Gets Created)

```
After running setup_dirs.py and create_dummy_dataset.py:

data/
├── foggy_cityscapes/
│   └── FC01/
│       ├── images/
│       │   ├── train/  ← 20 synthetic images generated here
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/

weights/  ← (will contain trained models after train.py)
results/  ← (will contain output images)
```

---

## ✅ SUCCESS INDICATORS

When you see these, everything works:

```
✅ Terminal shows: "You can now view your Streamlit app in your browser"
✅ URL appears: "Local URL: http://localhost:8501"
✅ Browser opens automatically
✅ You can upload images and get detections
✅ Detection results show in web interface
```

---

## 💾 FILES NEEDED FROM YOU

**To send your friend:**

```
DON'T SEND:           SEND IF FRIEND ASKS:
- Nothing!            - data.zip (47MB) for real images
                      - weights.zip (476MB) for pre-trained model
```

**How to create:**
```bash
# Windows PowerShell
Compress-Archive -Path "data" -DestinationPath "data.zip"
Compress-Archive -Path "weights" -DestinationPath "weights.zip"

# Mac/Linux
zip -r data.zip data/
zip -r weights.zip weights/
```

---

## 🚨 COMMON ISSUES & FIXES

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | `pip install torch torchvision` |
| `Command not found: streamlit` | `pip install streamlit` |
| `venv not activated` | Windows: `venv\Scripts\activate` / Mac: `source venv/bin/activate` |
| `No data found` | `python create_dummy_dataset.py` |
| Port 8501 already in use | `streamlit run app_streamlit.py --server.port 8502` |
| Slow/Out of memory | Normal on CPU. Use GPU or reduce batch size. |

---

## 📱 ONE-LINER SETUP (Copy & Paste Entire Block)

**Windows:**
```bash
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git && cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather\Foggy_Object_Detection && python -m venv venv && call venv\Scripts\activate.bat && pip install --upgrade pip && pip install -r requirements.txt && python setup_dirs.py && python create_dummy_dataset.py && streamlit run app_streamlit.py
```

**Mac/Linux:**
```bash
git clone https://github.com/T-sashi-pavan/MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather.git && cd MSFFA-YOLO-Network-Multiclass-Object-Detection-for-Traffic-Investigations-in-Foggy-Weather/Foggy_Object_Detection && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && python setup_dirs.py && python create_dummy_dataset.py && streamlit run app_streamlit.py
```

---

## 🎓 SUPPORTED COMMANDS AFTER SETUP

```bash
# Activate environment (ALWAYS first!)
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Web Interface
streamlit run app_streamlit.py

# Training
python train.py

# Testing
python main.py
python test_restoration.py
python test_full_pipeline.py
python test_on_real_data.py

# Utilities
python setup_dirs.py
python create_dummy_dataset.py
python analyze_dataset.py

# Deactivate environment (when done)
deactivate
```

---

**Save this file and send to your friend!**

---

*Last Updated: April 2026*
*For: Copy & Paste Setup*
