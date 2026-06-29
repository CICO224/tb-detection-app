"""
Tuberculosis Detection Backend API
==================================

FastAPI backend for TB detection from Chest X-rays.
Handles image preprocessing and model inference.

IMPORTANT MEDICAL DISCLAIMER:
This is a Decision Support Tool and NOT a final medical diagnosis.
The model may produce false positives and false negatives.
Always consult qualified medical professionals for actual diagnosis.
"""

import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import joblib

# Configuration
IMG_SIZE = 128  # Size for sklearn models (linear regression uses 128x128)
THRESHOLD = 0.5  # Probability threshold for classification
MODEL_TYPE = 'sklearn'  # 'sklearn' for .pkl files

# Initialize FastAPI app
app = FastAPI(
    title="TB Detection API",
    description="API for Tuberculosis detection from Chest X-rays (Decision Support Tool)",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store loaded model
model = None

def load_model():
    """
    Loads the trained TB detection model (sklearn format).
    Supports Logistic Regression and other sklearn models saved as .pkl
    """
    global model
    if model is None:
        model_path = 'tb_linear_model.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                "Please train the model first using train_model.py"
            )
        
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    return model

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesses an image for sklearn model inference.
    
    For sklearn models (Logistic Regression):
    - Convert to grayscale
    - Resize to IMG_SIZE x IMG_SIZE
    - Flatten to 1D array
    - Normalize pixel values to [0, 1]
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed image array ready for sklearn model input (1D array)
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Flatten the 2D image array into a 1D feature vector
    # Example: 128x128 becomes a single row of 16,384 pixels
    img_flattened = img_array.flatten()
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_flattened.astype('float32') / 255.0
    
    # Reshape for sklearn (needs 2D array even for single sample)
    # Shape: (1, n_features)
    img_reshaped = img_normalized.reshape(1, -1)
    
    return img_reshaped

def predict_image(file: UploadFile) -> dict:
    """
    Predicts TB from an uploaded chest X-ray image using sklearn model.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Dictionary containing prediction results:
        - probability: float (0-1)
        - label: str ("Normal" or "TB")
        - status: str ("success" or "error")
        - message: str (optional error message)
    """
    try:
        # Read image file
        file.file.seek(0)
        image_bytes = file.file.read()
        
        # Validate image bytes
        if len(image_bytes) == 0:
            raise ValueError("Empty file received")
        
        # Open image with PIL
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify it's a valid image
            image.verify()
            # Reopen after verify (verify closes the image)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as img_error:
            raise ValueError(f"Invalid image file: {str(img_error)}")
        
        # Preprocess image for sklearn model
        processed_image = preprocess_image(image)
        
        # Load model if not already loaded
        sklearn_model = load_model()
        
        # Make prediction using sklearn model
        try:
            # Get prediction probabilities
            # For binary classification: returns array([[prob_normal, prob_tb]])
            prediction_proba = sklearn_model.predict_proba(processed_image)
            
            # Extract probability for TB class (class 1)
            probability_tb = float(prediction_proba[0][1])
            
            # Get the predicted class
            prediction = sklearn_model.predict(processed_image)
            predicted_class = int(prediction[0])
            
        except AttributeError:
            # If model doesn't have predict_proba, use decision_function or predict
            raise RuntimeError(
                "Model must support predict_proba() method. "
                "Use LogisticRegression or other probability-based classifiers."
            )
        except Exception as pred_error:
            raise RuntimeError(f"Prediction failed: {str(pred_error)}")
        
        # Determine label based on prediction
        # 0 = Normal, 1 = TB
        label = "TB" if predicted_class == 1 else "Normal"
        
        return {
            "probability": round(probability_tb, 4),
            "label": label,
            "status": "success",
            "raw_probability": round(probability_tb, 4)
        }
        
    except Exception as e:
        return {
            "probability": 0.0,
            "label": "Error",
            "status": "error",
            "message": str(e)
        }

@app.on_event("startup")
async def startup_event():
    """
    Load model on application startup.
    """
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"WARNING: {str(e)}")
        print("The API will fail until the model is trained using: python train_model.py")

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "TB Detection API",
        "status": "running",
        "note": "This is a Decision Support Tool, not a medical diagnosis."
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint for TB detection.
    
    Accepts image files (JPG, PNG) and returns prediction results.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG, PNG, etc.)"
            )
        
        # Reset file pointer to beginning (sync operation)
        file.file.seek(0)
        
        # Make prediction
        result = predict_image(file)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Prediction failed")
            )
        
        # Add disclaimer to response
        result["disclaimer"] = (
            "This is a Decision Support Tool and NOT a final medical diagnosis. "
            "The model may produce false positives and false negatives. "
            "Always consult qualified medical professionals for actual diagnosis."
        )
        
        return result
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error in /predict endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

