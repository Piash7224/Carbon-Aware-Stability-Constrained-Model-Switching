# ==========================================
# 0. INSTALL DEPENDENCIES
# ==========================================
#!pip install torch_pruning codecarbon

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50
import torch_pruning as tp
from codecarbon import EmissionsTracker
import time
import copy
import os

# ==========================================
# 1. CONFIGURATION (CHECK PATHS!)
# ==========================================
# ---------------------------------------------------------
# RUN 1: Create Medium
#TARGET_NAME = 'resnet50_med'
#PRUNING_RATIO = 0.2          # Remove 20% of channels

# RUN 2 (Run this script again later): Create Light
TARGET_NAME = 'resnet50_light'
PRUNING_RATIO = 0.4        # Remove 40% of channels
# ---------------------------------------------------------

# IMPORTANT: Point this to your saved Baseline Model
BASE_MODEL_PATH = '/kaggle/input/family-of-model/models/resnet50_base_best.pth' 

EPOCHS = 8                   # 8 Epochs for healing (Extended)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001       # Low LR for fine-tuning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = '/kaggle/input/food41/images'

# ==========================================
# 2. DATA LOADERS (STRICT MATCH TO BASELINE)
# ==========================================
print("📂 Preparing Data...")

# A. Define Transforms
# --------------------
# TRAIN: Augmented (for Fine-Tuning)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# EVAL: Clean (for Validating/Testing)
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# B. Load Data & Create Clean Split
# ---------------------------------
try:
    # 1. Load Dataset TWICE (References same data, different transforms)
    full_data_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_data_eval  = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)
    
    # 2. Calculate Split Sizes
    total_count = len(full_data_train)
    train_count = int(0.70 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count
    
    # 3. Generate Indices (Seed 42 ensures EXACT match to Baseline Split)
    torch.manual_seed(42)
    dummy_split = random_split(range(total_count), [train_count, val_count, test_count])
    
    train_idx = dummy_split[0].indices
    val_idx   = dummy_split[1].indices
    test_idx  = dummy_split[2].indices

    # 4. Create Subsets with CORRECT Transforms
    train_set = Subset(full_data_train, train_idx) # Augmented
    val_set   = Subset(full_data_eval, val_idx)    # Clean
    test_set  = Subset(full_data_eval, test_idx)   # Clean

    # 5. Loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Data Split Created (Matches Baseline):")
    print(f"   - Train: {len(train_set)} (Fine-Tuning)")
    print(f"   - Val:   {len(val_set)} (Validation)")
    print(f"   - Test:  {len(test_set)} (Final Eval)")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")

# ==========================================
# 3. LOAD & PRUNE
# ==========================================
print(f"🛠️ Loading Base Model from {BASE_MODEL_PATH}...")

# 1. Load Architecture
model = resnet50(weights=None)
model.fc = nn.Linear(2048, 101) 

# 2. Load Weights (Safe loading)
if os.path.exists(BASE_MODEL_PATH):
    state_dict = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
else:
    raise FileNotFoundError(f"❌ Could not find {BASE_MODEL_PATH}. Check the path!")

# 3. Define Pruning Strategy
example_inputs = torch.randn(1, 3, 224, 224).to(DEVICE)
imp = tp.importance.MagnitudeImportance(p=1) 

ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 101:
        ignored_layers.append(m) # Don't prune the final classification layer

pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    iterative_steps=1, # One-shot pruning
    ch_sparsity=PRUNING_RATIO,
    ignored_layers=ignored_layers,
)

# 4. Execute Pruning
print(f"✂️ Pruning {PRUNING_RATIO*100}% of the model...")
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
pruner.step()
new_macs, new_nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(f"   📉 Params: {base_nparams/1e6:.2f}M -> {new_nparams/1e6:.2f}M")
print(f"   📉 FLOPs:  {base_macs/1e9:.2f}G -> {new_macs/1e9:.2f}G")

# ==========================================
# 4. FINE-TUNE (HEALING PROCESS)
# ==========================================
tracker = EmissionsTracker(
    project_name=f"FineTune_{TARGET_NAME}",
    output_dir="/kaggle/working/",
    measure_power_secs=15
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# CHANGED: Step size 3 allows for smoother healing over 8 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print(f"🚀 Starting Fine-Tuning ({EPOCHS} Epochs)...")
tracker.start()
start_time = time.time()

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training Step
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
    
    train_acc = correct.double() / len(train_set)
    
    # Validation Step (Clean Data)
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            
    val_acc = val_correct.double() / len(val_set)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"{TARGET_NAME}_best.pth")
        print(f"   💾 Best Model Saved ({best_acc:.4f})")
        
    scheduler.step()

emissions = tracker.stop()
tuning_time = time.time() - start_time

# ==========================================
# 5. FINAL TEST EVALUATION
# ==========================================
print("\n" + "="*40)
print("🔒 UNLOCKING TEST SET FOR FINAL EVALUATION")
print("="*40)

# Load the best fine-tuned model
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
print(f"⏱️ Tuning Time: {tuning_time/60:.2f} mins")
print(f"💨 Tuning Emissions: {emissions:.6f} kg CO2eq")
print("="*40)