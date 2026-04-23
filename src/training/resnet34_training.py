# ==========================================
# 0. INSTALL DEPENDENCIES
# ==========================================
#!pip install codecarbon

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet34, ResNet34_Weights
import time
import copy
import os
from codecarbon import EmissionsTracker

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = 'resnet34_base' 

EPOCHS = 25               # Consistency for fair comparison
PATIENCE = 5              # Early Stopping Patience
BATCH_SIZE = 128
LEARNING_RATE = 0.001     
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/kaggle/input/food41/images' 

# ==========================================
# 2. SETUP CARBON TRACKER
# ==========================================
tracker = EmissionsTracker(
    project_name=f"Food101_Training_{MODEL_NAME}",
    output_dir="/kaggle/working/",
    measure_power_secs=15
)

# ==========================================
# 3. DATA PREPARATION (FIXED: CLEAN EVAL)
# ==========================================
print("📂 Preparing Data...")

# A. Define Transforms
# --------------------
# TRAIN: Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# EVAL (Val/Test): Clean Center Crop
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# B. Load Data & Create Clean Split
# ---------------------------------
try:
    # 1. Load Dataset TWICE
    full_data_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_data_eval  = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)
    
    # 2. Calculate Split Sizes
    total_count = len(full_data_train)
    train_count = int(0.70 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count
    
    # 3. Generate Indices (Seed 42)
    torch.manual_seed(42)
    dummy_split = random_split(range(total_count), [train_count, val_count, test_count])
    
    train_idx = dummy_split[0].indices
    val_idx   = dummy_split[1].indices
    test_idx  = dummy_split[2].indices

    # 4. Create Subsets
    train_set = Subset(full_data_train, train_idx) # Augmented
    val_set   = Subset(full_data_eval, val_idx)    # Clean
    test_set  = Subset(full_data_eval, test_idx)   # Clean

    # 5. Loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Data Split Created (Verified Clean Eval):")
    print(f"   - Train: {len(train_set)} (Augmented)")
    print(f"   - Val:   {len(val_set)} (Clean)")
    print(f"   - Test:  {len(test_set)} (Clean)")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")

# ==========================================
# 4. MODEL SETUP (RESNET34)
# ==========================================
def get_model():
    print(f"🛠️ Building ResNet34...")
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    
    # ResNet34 has 512 input features in the final FC layer (same as ResNet18)
    # We replace it to output 101 classes
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model.to(DEVICE)

model = get_model()

# ==========================================
# 5. TRAINING LOOP (WITH EARLY STOPPING)
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step Size 7 to prevent underfitting
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(f"🚀 Starting Training for {EPOCHS} epochs (Patience: {PATIENCE})...")
tracker.start()

best_acc = 0.0
patience_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # TRAIN
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
    epoch_acc = correct.double() / len(train_set)
    
    # VALIDATION
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            
    val_acc = val_correct.double() / len(val_set)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    # CHECK FOR IMPROVEMENT
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        patience_counter = 0 
        print(f"   💾 Best Model Saved! ({best_acc:.4f})")
    else:
        patience_counter += 1
        print(f"   ⏳ No improvement. Patience: {patience_counter}/{PATIENCE}")
        
    # EARLY STOPPING
    if patience_counter >= PATIENCE:
        print(f"🛑 Early Stopping Triggered after {epoch+1} epochs.")
        break
    
    scheduler.step()

# STOP TRACKING
emissions = tracker.stop()
training_time = time.time() - start_time

# ==========================================
# 6. FINAL TEST EVALUATION
# ==========================================
print("\n" + "="*40)
print("🔒 UNLOCKING TEST SET FOR FINAL EVALUATION")
print("="*40)

model.load_state_dict(best_model_wts)
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)
        test_total += labels.size(0)

test_acc = test_correct.double() / test_total

print(f"🏆 Final Test Accuracy: {test_acc:.4f}")
print(f"⏱️ Training Time: {training_time/60:.2f} mins")
print(f"💨 Carbon Emissions: {emissions:.6f} kg CO2eq")
print("="*40)