# Tuberculosis Detection System - Complete Documentation

## Deploy as a Public Link

The app is ready for a one-link Streamlit deployment. The Streamlit UI now runs the model directly, so you do not need to deploy the separate FastAPI backend.

1. Push this project folder to a GitHub repository.
2. Make sure these files are included in the repo:
   - `app_ui.py`
   - `app_model.py`
   - `requirements.txt`
   - `runtime.txt`
   - `tb_linear_model.pkl`
   - `.streamlit/config.toml`
3. Go to `https://share.streamlit.io/`.
4. Select **New app**.
5. Choose your GitHub repository and branch.
6. Set the main file path to `app_ui.py`.
7. Deploy.

After deployment, Streamlit will give you a public URL that you can share.

A local MVP (Minimum Viable Product) for Tuberculosis detection from Chest X-rays using Machine Learning with Logistic Regression (Linear Model).

**Project Team**: Bayuputra Kurnia Adhyatma, Bryan Vincent, Richson Limec

---

## ⚠️ IMPORTANT WARNINGS - READ THIS FIRST

### Critical Errors You May Encounter

**Before starting, be aware of these common issues:**

1. **NumPy Compatibility Issue**
   - **Error**: `AttributeError: _ARRAY_API not found` or `numpy.core.multiarray failed to import`
   - **Cause**: OpenCV 4.8.1.78 is compiled for NumPy 1.x, but NumPy 2.x got installed
   - **Solution**: Lock NumPy to version 1.x in requirements.txt: `numpy<2`

2. **Missing Modules**
   - **Error**: `ModuleNotFoundError: No module named 'matplotlib'` or `seaborn`
   - **Cause**: Dependencies missing from requirements.txt or not installed
   - **Solution**: Run `pip install -r requirements.txt` to install all required packages

3. **Virtual Environment Not Activated**
   - **Error**: Packages install to system Python instead of venv
   - **Solution**: Always run `.\venv\Scripts\activate` before installing

4. **Backend Port Already in Use**
   - **Error**: `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`
   - **Cause**: Another process is using port 8000
   - **Solution**: Run `Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force` to free the port

5. **VS Code Import Warnings**
   - **Error**: Pylance shows "Import could not be resolved"
   - **Cause**: VS Code not using venv Python interpreter
   - **Solution**: Select interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` → Choose `.\venv\Scripts\python.exe`

6. **Backend Connection Issues**
   - **Error**: "Backend API: Not Running" in frontend
   - **Cause**: Backend not started or crashed
   - **Solution**: Check backend terminal for errors, ensure port 8000 is available

---

## ⚠️ MEDICAL DISCLAIMER

**This is a Decision Support Tool and NOT a final medical diagnosis.**

- The model may produce **false positives** and **false negatives**
- This system is for **educational and research purposes only**
- **Always consult qualified medical professionals** for actual diagnosis
- **Not approved for clinical use**

---

## 📋 Table of Contents

1. [How the Application Works](#how-the-application-works)
2. [AI Training Methods](#ai-training-methods)
3. [Installation Guide](#installation-guide)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)
6. [Technical Details](#technical-details)

---

## 🧠 How the Application Works

### System Architecture

The application consists of three main components:

#### 1. **Frontend (Streamlit UI)** - `app_ui.py`
- **Purpose**: User interface for uploading X-ray images and viewing results
- **Technology**: Streamlit 1.18.0
- **Features**:
  - Drag-and-drop image upload
  - Real-time image preview
  - Color-coded results (Green = Normal, Red = TB)
  - Detailed explanations of predictions
  - System status monitoring

#### 2. **Backend API (FastAPI)** - `app_backend.py`
- **Purpose**: Handles image processing and model inference
- **Technology**: FastAPI 0.88.0, Uvicorn 0.20.0
- **Process**:
  1. Receives uploaded image from frontend
  2. Preprocesses image (resize to 128×128, convert to grayscale, normalize pixels)
  3. Loads trained Logistic Regression model (.pkl file)
  4. Flattens image to 1D feature vector and runs inference
  5. Returns JSON response with prediction and confidence

#### 3. **AI Model (Linear Model)** - `train_model.py`
- **Purpose**: Trains the Logistic Regression model
- **Technology**: scikit-learn (sklearn)
- **Output**: `tb_linear_model.pkl` - trained model weights

### Prediction Workflow

```
User uploads X-ray
    ↓
Frontend sends to Backend API (POST /predict)
    ↓
Backend preprocesses image:
  - Resize to 128×128 pixels
  - Convert to grayscale
  - Normalize pixel values (0-255 → 0-1)
    ↓
Model inference:
  - Flatten 128×128 image into 16,384 features
  - Pass through Logistic Regression model
  - Outputs probability (0.0 to 1.0)
    ↓
Classification:
  - If probability ≥ 0.5 → "TB"
  - If probability < 0.5 → "Normal"
    ↓
Backend returns JSON:
  {
    "label": "TB" or "Normal",
    "probability": 0.0 to 1.0,
    "status": "success"
  }
    ↓
Frontend displays result with explanation
```

### Why the AI Predicts Normal or TB

The AI makes predictions based on **learned visual patterns**, not database lookups:

**For Normal Prediction:**
- Detects clear lung fields (no opacities)
- Identifies normal lung markings (blood vessels)
- Sees no consolidation (solid white areas)
- Finds no cavities (dark holes)
- Recognizes normal anatomical structures

**For TB Prediction:**
- Detects abnormal opacities (cloudy/white areas)
- Identifies possible consolidation (infection signs)
- Finds potential cavities (TB lesions)
- Recognizes upper lobe patterns (common TB location)
- Sees pleural changes (thickening/fluid)

**Important**: The model doesn't store images. Each prediction is a fresh analysis through the neural network based on patterns learned during training.
Logistic Regression classifier
---

## 🎓 AI Training Methods

### Model Architecture

**Logistic Regression (Linear Model)**:

Logistic Regression is a simple yet effective linear classification algorithm. The model works as follows:

```
Input: Flattened X-ray image (16,384 features = 128×128 grayscale pixels)
    ↓
Linear Combination:
  z = w₁×pixel₁ + w₂×pixel₂ + ... + w₁₆₃₈₄×pixel₁₆₃₈₄ + bias
    ↓
Sigmoid Function (Logistic Function):
  probability = 1 / (1 + e^(-z))
  Converts z to a probability between 0 and 1
    ↓
Output: Binary classification
  - p ≥ 0.5 → TB (class 1)
  - p < 0.5 → Normal (class 0)
```

**Model Characteristics**:
- **Parameters**: ~2.1 million (one weight per pixel feature + bias)
- **Model Size**: ~5-10 MB (pickle format)
- **Training Time**: Fast (minutes to tens of minutes)
- **Inference Time**: Very fast (milliseconds per image)
- **Memory Usage**: Lower than deep learning (entire dataset loaded into RAM)

**Advantages**:
- Simple, interpretable model
- Fast training and inference
- Low memory footprint
- Good for CPU-only environments

**Limitations**:
- May not capture complex non-linear patterns
- Performance depends heavily on image features
- Sensitive to image preprocessing

### Training Process

#### 1. **Data Preparation**
- Images organized into `data/train/` directory
- Each folder contains `Normal/` and `TB/` subfolders
- Recommended: 500+ images per class for good accuracy
- Images automatically resized to 128×128 and converted to grayscale during training

#### 2. **Image Preprocessing**
For Linear Models:
- **Resize**: 128×128 pixels (smaller than CNN to fit in memory)
- **Grayscale**: Convert RGB to single-channel (reduces features)
- **Flatten**: 128×128 → 16,384 features (1D vector)
- **Normalize**: Pixel values scaled from 0-255 to 0-1

#### 3. **Training Configuration**

**Hyperparameters:**
- **Model**: LogisticRegression
- **Max Iterations**: 2000
- **Class Weight**: Balanced (handles imbalanced datasets)
- **Solver**: LBFGS (default)
- **Random State**: 42 (reproducibility)
- **N Jobs**: -1 (uses all CPU cores)

**Train-Test Split**:
- Configurable percentage (default: 75% train, 25% test)
- Stratified split ensures equal class distribution in both sets
- Can be adjusted in `running.ipynb` cell 3

#### 4. **Training Output**
- **Model File**: `tb_linear_model.pkl` (saved as Python pickle)
- **Test Results**: CSV file with predictions and actual labels
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Visualization showing TP, TN, FP, FN

### Training Time Estimates

- **Training**: 1-5 minutes (CPU, depends on dataset size)
- **Testing**: <1 second (evaluation on test set)
- **Inference**: 10-50ms per image

---

## 🚀 Installation Guide

### Prerequisites

- **Python 3.8 or higher** (tested with Python 3.11.5)
- **Windows 10/11**
- **pip** (Python package manager)
- **8GB+ RAM recommended** (for training)
- **5GB+ free disk space** (for dependencies and model)

### Step 1: Navigate to Project Directory
11.x** (tested with Python 3.11.5)
- **Windows 10/11**
- **pip** (Python package manager)
- **4GB+ RAM** (for training)
- **2GB+ free disk space** (for dependencies and model)

### Step 1: Navigate to Project Directory

```powershell
cd d:\AI\ Project
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Expected installation time**: 3-8 minutes

**What gets installed:**
- `tensorflow==2.15.0` - (TensorFlow for compatibility, though not actively used in linear model)
- `scikit-learn==1.3.2` - Linear model and machine learning utilities
- `fastapi==0.88.0` - Backend API framework
- `uvicorn[standard]==0.20.0` - ASGI server
- `streamlit==1.18.0` - Frontend UI framework
- `numpy==1.24.3` - Numerical computing (NumPy < 2 for OpenCV compatibility)
- `opencv-python==4.8.1.78` - Image processing
- `Pillow==10.0.0` - Image processing
- `matplotlib==3.7.2` - Plotting and visualization
- `seaborn==0.13.0` - Statistical visualization
- `scipy==1.11.4` - Scientific computing
- `requests==2.31.0` - HTTP client
- `python-multipart==0.0.6` - File upload support
- `joblib` - Modelsklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import streamlit; print('Streamlit OK')"
python -c "import fastapi; print('FastAPI OK')"
```

### Step 6: Prepare Training Data

Organize your chest X-ray images:

```
data/
├── train/
│   ├── Normal/    (place normal X-ray images here)
│   └── TB/        (place TB-positive X-ray images here)
└── test/
    ├── Normal/    (place normal X-ray images here - for reference)
    └── TB/        (place TB-positive X-ray images here - for reference)
```

**Note**: The linear model training will automatically split the train folder into train/test sets. The test folder structure is for reference or when you need separate test data.

**Recommended dataset size:**
- Minimum: 100 images per class
- Good: 300+ images per class
- Optimal: 500+ images per class

**Image requirements:**
- Formats: JPG, JPEG, PNG
- Size: Any (automatically resized to 128×128)
- Quality: Clear, properly labeled images
**Helper script** (if your data is in a different structure):
```powershell
python organize_data.py --source "C:\path\to\your\images\folder"
```

---

## ▶️ Running the Application

### Option 1: Using a Pre-trained Model
linear_model.pkl`:

#### Terminal 1: Start Backend
```powershell
cd d:\AI\ Project
.\venv\Scripts\activate
python app_backend.py
```

You should see:
```
Loading model from tb_linear_model.pkl...
Model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!**

#### Terminal 2: Start Frontend
```powershell
cd d:\AI\ Project
.\venv\Scripts\activate
streamlit run app_ui.py
```

The browser will open automatically at `http://localhost:8501`

### Option 2: Train Your Own Model First

#### Step 1: Train the Model in Jupyter

Two ways to train:

**Method A: Using Jupyter Notebook** (Recommended for visualization)
```powershell
cd d:\AI\ Project
.\venv\Scripts\activate
jupyter notebook running.ipynb
```

Then run the cells in order:
1. Install dependencies
2. Import libraries
3. Set configuration
4. Load data
5. Split data
6. Train model
7. Evaluate model
8. Export results to CSV

**Method B: Using Python Script** (Faster)
```powershell
cd d:\AI\ Project
.\venv\Scripts\activate
python train_model.py
```

Wait for training to complete. You'll see:
- Training progress with metrics
- Final test results (accuracy, precision, recall, F1-score)
- Confusion matrix visualization
- Model saved as `tb_linear_model.pkl`
- Results exported to `tb_detection_results.csv`

#### Step 2: Start Backend and Frontend
Follow steps from Option 1 above.
NumPy Compatibility Error
**Symptoms**: `AttributeError: _ARRAY_API not found` or `numpy.core.multiarray failed to import`

**Solution**:
Ensure NumPy is locked to version 1.x:
```powershell
pip install "numpy<2"
pip install -r requirements.txt
```

#### Issue: Missing matplotlib or seaborn
**Symptoms**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**:
```powershell
pip install matplotlib==3.7.2
pip install seaborn==0.13.0
```

#### Issue: "ResolutionImpossible" - Dependency Conflicts
**Symptoms**: `ERROR: ResolutionImpossible` when installing

**Solution**:
```powershell
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### Issue: Packages Installing to System Python
**Symptoms**: Packages install to `C:\Users\...\AppData\...` instead of venv

**Solution**:
```powershell
# Verify venv is activated (should see (venv) in prompt)
.\venv\Scripts\activate

# Verify using venv Python
python -c "import sys; print(sys.executable)"
# Should show: C:\...\venv\Scripts\python.exe

# Then install
pip install -r requirements.txt
```

### Training Issues

#### Issue: "Out of Memory" During Training
**Symptoms**: Training crashes with memory error

**Solution**:
1. Reduce batch size (if using train_model.py)
2. Close other applications
3. Use fewer training images
4. Reduce image size from 128×128 to 64×64

#### Issue: Training Takes Too Long
**Solution**:
- Early stopping will automatically stop if no improvement
- Use a smaller dataset for testing
- Linear models should train in minutes, not hours

### Runtime Issues

#### Issue: "Model file not found"
**Symptoms**: Backend fails to start

**Solution**:
1. Train the model first: Run `train_model.py` or use Jupyter notebook
2. Verify `tb_linear_model.pkl` exists in project folder
3. Check backend terminal for specific error message

#### Issue: Port 8000 Already in Use
**Symptoms**: `[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`

**Solution**:
```powershell
# Kill the process using port 8000
Get-NetTCPConnection -LocalPort 8000 | Stop-Process -Force

# Then restart backend
python app_backend.py
```

#### Issue: "Backend API: Not Running"
**Symptoms**: Frontend shows "Backend API: Not Running"

**Solution**:
1. Check backend terminal - is it running?
2. Check port 8000 is free
3. Restart backend: Stop (Ctrl+C) and run `python app_backend.py` again
4. Check Windows Firewall - allow when prompted

#### Issue: Image Upload Fails
**Symptoms**: "Error loading image" or file won't upload

**Solution**:
1. Check file format: Must be JPG, JPEG, or PNG
2. Check file size: Should be < 50MB
3. Try a different image
4. Check backend terminal for errors

### VS Code Issues

#### Issue: Import Warnings in VS Code
**Symptoms**: Red squiggly lines under imports

**Solution**:
1. Select correct Python interpreter:
   - Press `Ctrl+Shift+P`
   - Type: `Python: Select Interpreter`
   - Choose: `.\venv\Scripts\python.exe`
2. Reload VS Code: `Ctrl+Shift+P` → `Developer: Reload Window`

**Note**: These are just warnings. Code will run fine from terminal.
   - Press `Ctrl+Shift+P`
   - Type: `Python: Select Interpreter`
   - Choose: `.\venv\Scripts\python.exe`
2. Reload VS Code: `Ctrl+Shift+P` → `Developer: Reload Window`
3. Create `.vscode/settings.json`:
   ```json
   {
       "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
       "python.terminal.activateEnvironment": true
   }
   ```

**Note**: These are just warnings. Code will run fine from terminal even if VS Code shows warnings.

#### Issue: "verbose=1" Type Errors
**Symptoms**: Pylance complains about verbose parameter

**Solution**: Already fixed with `# type: ignore` comments. If still seeing errors:
- Reload VS Code window
- The code will run fine - these are just type checking warnings

---

## 📊 Technical Details

### Version Information

**Python**: 3.11.x (tested with 3.11.5)

**Key Dependencies**:
- `scikit-learn==1.3.2` (Machine Learning Library)
- `fastapi==0.88.0` (Backend API)
- `uvicorn[standard]==0.20.0` (ASGI Server)
- `streamlit==1.18.0` (Frontend)
- `numpy==1.24.3` (NumPy < 2 for OpenCV compatibility)
- `opencv-python==4.8.1.78` (Image Processing)
- `Pillow==10.0.0` (Image Processing)
- `matplotlib==3.7.2` (Visualization)
- `seaborn==0.13.0` (Statistical Visualization)
- `scipy==1.11.4` (Scientific Computing)

### Model Specifications

- **Algorithm**: Logistic Regression (Linear Classification)
- **Input Size**: 128×128×1 (grayscale) → 16,384 features
- **Model Parameters**: ~2.1 million
- **Output**: Binary classification (Normal/TB probability)
- **Optimization**: LBFGS solver
- **Regularization**: L2 (ridge), class weights for imbalance

### Performance Metrics

- **Training Accuracy**: Typically 80-95% (depends on dataset)
- **Inference Time**: 10-50ms per image (CPU)
- **Model Size**: ~5-10 MB (`tb_linear_model.pkl`)
- **Memory Usage**: ~500MB-1GB during training

### API Endpoints

**Backend API** (`http://localhost:8000`):

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /predict` - Upload image and get prediction
  - **Request**: Multipart form data with `file` field
  - **Response**: JSON with `label`, `probability`, `status`

### File Structure

```
AI Project/
├── train_model.py                # Linear model training script
├── app_backend.py                # FastAPI backend server
├── app_ui.py                     # Streamlit frontend
├── running.ipynb                 # Jupyter notebook (interactive training)
├── organize_data.py              # Helper to organize training data
├── verify_setup.py               # Setup verification script
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
├── tb_linear_model.pkl           # Trained model (generated)
├── tb_detection_results.csv      # Test results (generated)
├── training_history.csv          # Training metrics (generated)
├── confusion_matrix.png          # Visualization (generated)
└── data/                         # Training and test data
    ├── train/
    │   ├── Normal/
    │   └── TB/
    └── test/
        ├── Normal/
        └── TB/
```

### Model Training Pipeline

1. **Data Loading**: Images read from `data/train/` directory
2. **Preprocessing**: Resize to 128×128, convert to grayscale
3. **Flattening**: 2D images converted to 1D feature vectors
4. **Train-Test Split**: Stratified split (default 75-25)
5. **Model Training**: Logistic Regression with balanced class weights
6. **Evaluation**: Metrics calculated on test set
7. **Visualization**: Confusion matrix and results saved
8. **Export**: Results exported to CSV for analysis

---

## 📚 Additional Resources

- **Scikit-Learn Documentation**: https://scikit-learn.org/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **OpenCV Documentation**: https://docs.opencv.org/

---

## 📄 License

This project is for **educational and research purposes only**. Not for clinical use.

---

## 👨‍💻 Development Notes

**Technology Stack**:
- **Backend**: Python (FastAPI 0.88.0, Uvicorn 0.20.0)
- **Frontend**: Streamlit 1.18.0
- **ML Model**: scikit-learn (Logistic Regression)
- **Image Processing**: OpenCV, PIL/Pillow
- **Scientific Computing**: NumPy, SciPy
- **Training Interface**: Jupyter Notebook

**Key Design Decisions**:
- **Linear Model**: Simple, interpretable, fast training and inference
- **Grayscale Images**: Reduces dimensionality (3 channels → 1 channel)
- **Smaller Input Size**: 128×128 (vs 224×224) to fit in memory
- **Scikit-learn**: CPU-friendly, easy to deploy
- **Class Balancing**: Handles imbalanced datasets automatically
- **Train-Test Split**: Configurable percentage for flexibility
- **Jupyter Notebook**: Interactive training with visualization

**Project Files Summary**:
- `train_model.py` - Command-line training script
- `running.ipynb` - Interactive Jupyter notebook with train-test splitting
- `app_backend.py` - FastAPI server for predictions
- `app_ui.py` - Streamlit web interface
- `verify_setup.py` - Environment verification
- `organize_data.py` - Data organization helper

**Deployment Considerations**:
- Model saved as pickle (.pkl) for easy loading
- Lightweight and CPU-compatible for Windows deployment
- CORS enabled for frontend-backend communication
- Error handling for various edge cases
- CSV export for result analysis

---

**Remember**: This is a Decision Support Tool. Always consult qualified medical professionals for actual diagnosis.
