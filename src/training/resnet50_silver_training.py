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
# 1. OPTIMIZED CONFIGURATION
# ==========================================
TARGET_NAME = 'resnet50_silver_final'
PRUNING_RATIO = 0.35          # <--- The Magic Number (Silver Bullet)
EPOCHS = 15                   # <--- Enough to reach peak accuracy in one run
BATCH_SIZE = 64
LEARNING_RATE = 0.0001        # Start at 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATH TO YOUR ORIGINAL BASELINE
BASE_MODEL_PATH = '/kaggle/input/family-of-model/models/resnet50_base_best.pth' 
DATA_DIR = '/kaggle/input/food41/images'

# ==========================================
# 2. DATA SETUP
# ==========================================
print("📂 Preparing Data...")
# Training Transform (Augmentation)
train_transform = transforms.Compose([
    transforms.Resize(256), transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Eval Transform (Clean)
eval_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    full_data_train = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_data_eval  = datasets.ImageFolder(root=DATA_DIR, transform=eval_transform)
    
    total_count = len(full_data_train)
    train_count = int(0.70 * total_count)
    val_count = int(0.15 * total_count)
    test_count = total_count - train_count - val_count
    
    torch.manual_seed(42)
    dummy_split = random_split(range(total_count), [train_count, val_count, test_count])
    
    train_loader = DataLoader(Subset(full_data_train, dummy_split[0].indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(Subset(full_data_eval, dummy_split[1].indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(Subset(full_data_eval, dummy_split[2].indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Data Ready: Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")

# ==========================================
# 3. CREATE & PRUNE MODEL (0.35)
# ==========================================
print(f"🛠️ Creating Silver Model (Ratio: {PRUNING_RATIO})...")

# Load Baseline
model = resnet50(weights=None)
model.fc = nn.Linear(2048, 101)
if os.path.exists(BASE_MODEL_PATH):
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
else:
    raise FileNotFoundError("Check BASE_MODEL_PATH!")

# Prune
example_inputs = torch.randn(1, 3, 224, 224).to(DEVICE)
imp = tp.importance.MagnitudeImportance(p=1)
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 101:
        ignored_layers.append(m)

pruner = tp.pruner.MagnitudePruner(
    model, example_inputs, importance=imp, iterative_steps=1,
    ch_sparsity=PRUNING_RATIO,
    ignored_layers=ignored_layers,
)
pruner.step()

print(f"✅ Pruning Complete! Model is ready for training.")

# ==========================================
# 4. TRAINING (THE OPTIMIZED PART)
# ==========================================
tracker = EmissionsTracker(project_name=f"FineTune_{TARGET_NAME}", output_dir="/kaggle/working/")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🏆 THE FIX: Gentle Scheduler
# Instead of dropping every 3 epochs (too fast), we drop every 5.
# Instead of dropping by 90% (gamma 0.1), we drop by 50% (gamma 0.5).
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

print(f"🚀 Starting Training ({EPOCHS} Epochs)...")
tracker.start()
start_time = time.time()
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
    
    train_acc = correct.double() / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            
    val_acc = val_correct.double() / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"{TARGET_NAME}_best.pth")
        print(f"   💾 Saved Best Model ({best_acc:.4f})")
    
    scheduler.step()

emissions = tracker.stop()
tuning_time = time.time() - start_time

# ==========================================
# 5. FINAL TEST
# ==========================================
print("\n🔒 FINAL TEST EVALUATION")
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

print(f"🏆 Final SILVER Accuracy: {test_correct.double()/test_total:.4f}")
print(f"⏱️ Total Time: {tuning_time/60:.2f} mins")