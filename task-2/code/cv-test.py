import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# --- Configuration ---
# Note: I am keeping your modified relative paths here.
MODEL_PATH = '../dataset/cv_animal_model/resnet18_best.pth'
CLASSES_PATH = '../dataset/cv_animal_model/class_names.txt'
IMAGE_SIZE = 224 # Standard input size for ResNet

def load_classes(classes_path):
    """Loads class names from a text file."""
    print(f"[LOG] Starting class loading from: {classes_path}")
    if not os.path.exists(classes_path):
        print(f"Error: Class names file not found at {classes_path}")
        sys.exit(1)
    
    try:
        with open(classes_path, 'r') as f:
            # Assuming one class name per line, stripped of whitespace
            class_names = [line.strip() for line in f.readlines()]
        
        if not class_names:
            print(f"Error: {classes_path} is empty.")
            sys.exit(1)
            
        print(f"[LOG] Successfully loaded {len(class_names)} classes.")
        print(f"[LOG] First 3 classes: {class_names[:3]}")
        return class_names
    except Exception as e:
        print(f"[LOG] CRITICAL ERROR during class file reading: {e}")
        sys.exit(1)

def load_model(model_path, num_classes):
    """Initializes and loads the ResNet-18 model weights."""
    print(f"[LOG] Starting model loading process.")
    print(f"Loading model architecture (ResNet-18) for {num_classes} classes...")
    
    # Initialize the model structure
    model = models.resnet18(weights=None) 
    
    # Adjust the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"[LOG] Model structure created. Final layer configured for {num_classes} outputs.")
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}. Check the path.")
        sys.exit(1)

    # Load state dictionary
    try:
        print(f"[LOG] Attempting to load weights from: {model_path}")
        # Load weights onto the CPU
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check for DataParallel wrapper issue
        if list(state_dict.keys())[0].startswith('module.'):
            print("[LOG] Detected 'module.' prefix in state dict keys. Correcting...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval() # Set the model to evaluation mode
        print("[LOG] Model weights loaded and model set to evaluation mode.")
        return model
    except Exception as e:
        print(f"[LOG] CRITICAL ERROR loading model weights: {e}")
        sys.exit(1)

def get_image_transforms(image_size):
    """Defines the standard image preprocessing pipeline."""
    print(f"[LOG] Defining image transformation pipeline (Resize 256, Crop {image_size}).")
    # Standard normalization for ImageNet-trained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize(256),         
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),           
        normalize                        
    ])

def classify_image(image_path, model, class_names, data_transforms):
    """Loads an image, makes a prediction, and returns the class name."""
    print(f"\n[LOG] Starting classification for image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'. Check the path and file name.")
        return "Classification Failed (Image Not Found)"

    # 1. Load image
    try:
        print("[LOG] Attempting to open image...")
        image = Image.open(image_path).convert('RGB')
        print("[LOG] Image opened successfully.")
    except Exception as e:
        print(f"Error opening image: {e}")
        return "Classification Failed (Image Load Error)"
        
    # 2. Apply transformations
    try:
        print("[LOG] Applying transformations and converting to tensor...")
        input_tensor = data_transforms(image)
        # Add an extra dimension for batch size (1)
        input_batch = input_tensor.unsqueeze(0) 
        print(f"[LOG] Input batch shape: {input_batch.shape}")
    except Exception as e:
        print(f"Error during transformation: {e}")
        return "Classification Failed (Transformation Error)"

    # 3. Prediction
    try:
        print("[LOG] Performing forward pass (prediction)...")
        with torch.no_grad():
            output = model(input_batch)
        print("[LOG] Prediction complete. Calculating probabilities.")

        # Get probabilities and the prediction index
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_index = torch.max(probabilities, 0)
        
        # Map index to class name
        predicted_class = class_names[predicted_index.item()]
        confidence = probabilities[predicted_index.item()].item() * 100

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Classification Failed (Prediction Error)"

    print("-" * 30)
    print(f"Prediction Result:")
    print(f"  Class Name: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print("-" * 30)
    
    return predicted_class

def main():
    # print("[LOG] Script execution started.") # REMOVE this log to prevent STDOUT pollution
    
    # --- Input Handling: MODIFIED FOR SUBPROCESS ---
    image_path = None

    # Check for the required number of arguments (script name + 1 argument = 2)
    if len(sys.argv) == 2:
        image_path = sys.argv[1] # Image path is the first argument after the script name
        # print(f"[LOG] Image path received from command-line: {image_path}", file=sys.stderr) # Use sys.stderr for logs
    else:
        # If running as a subprocess, we exit if arguments are wrong
        print("\n[ERROR] Invalid number of arguments. Usage: python cv-test.py <path/to/image.jpg>", file=sys.stderr)
        sys.exit(1)
    # -----------------------------------------------

    try:
        # Step 1: Load classes
        class_names = load_classes(CLASSES_PATH)
        num_classes = len(class_names)

        # Step 2: Load model
        model = load_model(MODEL_PATH, num_classes)
        
        # Step 3: Define transforms
        data_transforms = get_image_transforms(IMAGE_SIZE)
        
        # Step 4: Classify - Get the raw class name
        predicted_class = classify_image(image_path, model, class_names, data_transforms)

        # Step 5: Output the result cleanly to STDOUT
        # CRITICAL: Print only the resulting class name, no extra characters or newlines.
        print(predicted_class, end='', flush=True)

    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred in cv-test.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
