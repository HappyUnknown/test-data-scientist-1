import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler # Import scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets.folder import default_loader
from PIL import Image
import os
import argparse
from pathlib import Path
from typing import List

# Removed: CV_FOLDER_PATH = "../dataset/cv_animal_model"

# configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Train an Image Classification Model for Animal Recognition.")
    
    # POSITIONAL ARGUMENT 1: Image Dataset Path
    parser.add_argument("data_dir_path", type=str, 
                        help="Mandatory Positional 1: Root directory containing animal class folders (e.g., Cat, Lion).")

    # POSITIONAL ARGUMENT 2: CV Model Output Path
    parser.add_argument("cv_folder_path", type=str, 
                        help="Mandatory Positional 2: Path to the directory where the trained model and class names will be saved (CV_FOLDER_PATH).")
    
    # Optional keyword arguments remain unchanged
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model architecture (e.g., resnet18).")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of training epochs. Increased to allow scheduler decay.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data to use for training.")
    
    args = parser.parse_args()
    
    # Map positional arguments to the variable names used in main()
    args.data_dir = args.data_dir_path
    args.output_dir = args.cv_folder_path
        
    return args

def safe_image_loader(path):
    """Safely loads an image, converts it to RGB,
    returning None if the file is corrupted"""
    try:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        # warning print without worker crash
        print(f"Warning: Failed to load image {path}. Skipping. Error: {e}", flush=True)
        return None

def main():
    args = parse_args()
    
    # set PyTorch multiprocessing context
    try:
        if torch.cuda.is_available() and torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        print(f"Warning: Could not set start method to 'spawn'. Proceeding. Error: {e}")

    # args.output_dir is set by the mandatory positional argument
    Path(args.output_dir).mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # data augmentation and loading
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    print(f"Attempting to load data from root: {args.data_dir}")
    
    try:
        # args.data_dir is now set by the first positional argument
        full_dataset = datasets.ImageFolder(args.data_dir, data_transforms, loader=safe_image_loader)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load ImageFolder from {args.data_dir}.")
        print("Please ensure your data structure is correct (data_dir/CLASS_NAME/image.jpg).")
        print(f"Original error: {e}")
        return

    total_size = len(full_dataset)
    if total_size == 0:
        print("CRITICAL ERROR: Dataset is empty. Check data_dir path and image contents.")
        return

    # split data
    train_size = int(args.train_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    image_datasets = {
        'train': train_dataset,
        'val': val_dataset
    }
    
    num_workers_to_use = 0 

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=num_workers_to_use),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=num_workers_to_use)
    }
    
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    # get class names from the underlying ImageFolder object
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Detected {num_classes} classes: {class_names}")
    print(f"Training on {train_size} images, validating on {val_size} images.")

    # model initialization
    if args.model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False
        
        # unfreeze the last convolutional block (layer4) and the final FC layer (fc)
        # allows the model to fine-tune high-level features for certain classes
        for param in model.layer4.parameters(): # unfreeze last block
            param.requires_grad = True
            
        # replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise NotImplementedError(f"Model {args.model_name} not yet implemented.")

    model = model.to(device)

    # loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    
    # only optimize parameters where requires_grad=True (layer4 and fc)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate
    )
    
    # decreases the learning rate
    # helps stabilize training and achieve better convergence.
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # training loop
    best_acc = 0.0
    
    print("\n" + "="*50)
    print("Starting Image Classification Fine-tuning...")
    print("="*50 + "\n")

    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch}/{args.num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            print(f"[{phase}] Starting data loader iteration for {phase} phase.", flush=True)

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # save the best model based on validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # save the model state dict
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_name}_best.pth"))
                
        # step the scheduler after the training phase of the epoch
        # it is safer to put this outside the inner loop
        if phase == 'val': # Check if validation phase just finished
            scheduler.step()
            print(f"Scheduler stepped. New learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
    # --- save class name list ---
    with open(os.path.join(args.output_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to {args.output_dir}/{args.model_name}_best.pth")


if __name__ == "__main__":
    main()