# ======================================================
# CIFAR-10 FINE TUNING FOR PRUNED MODELS
# DATE: 30-01-2026
# HARDENED AGAINST CUDA ECC FAILURES
# ======================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import random

# ======================================================
# PATHS
# ======================================================
DATA_DIR = "/kaggle/input/cifar10-pngs-in-folders/cifar10"

MEDIUM_MODEL_PATH = "/kaggle/input/pruned-models/pruned/resnet50_med_best.pth"
SILVER_MODEL_PATH = "/kaggle/input/pruned-models/pruned/resnet50_silver_final_best.pth"
LIGHT_MODEL_PATH  = "/kaggle/input/pruned-models/pruned/resnet50_light_RESCUED.pth"

SAVE_DIR = "/kaggle/working/fine_tuned_pruned"

# ======================================================
# HYPERPARAMETERS
# ======================================================
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 25
STAGE2_EPOCH = 8
EARLY_STOPPING_PATIENCE = 5
LR_FC = 3e-3
SEED = 42
MIXUP_ALPHA = 0.4

torch.manual_seed(SEED)
random.seed(SEED)

# ======================================================
# SAFE DEVICE SELECTION (CRITICAL FIX)
# ======================================================
def get_safe_device():
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available → Using CPU")
        return torch.device("cpu")

    try:
        _ = torch.tensor([1.0]).cuda()
        print("✅ CUDA OK")
        return torch.device("cuda")
    except Exception as e:
        print("❌ CUDA ERROR → Falling back to CPU")
        print(str(e))
        return torch.device("cpu")

# ======================================================
# DATASET
# ======================================================
def get_dataloaders(device):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    full_train = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len

    train_set, val_set = random_split(
        full_train, [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_set.dataset.transform = val_tf

    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), val_tf)

    return (
        DataLoader(train_set, BATCH_SIZE, shuffle=True,
                   num_workers=2, pin_memory=(device.type == "cuda")),
        DataLoader(val_set, BATCH_SIZE, shuffle=False,
                   num_workers=2, pin_memory=(device.type == "cuda")),
        DataLoader(test_set, BATCH_SIZE, shuffle=False,
                   num_workers=2, pin_memory=(device.type == "cuda")),
    )

# ======================================================
# MIXUP
# ======================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = random.betavariate(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================================================
# BUILD PRUNED MODEL (CPU SAFE)
# ======================================================
def build_pruned_model(name, path, device):
    print("🔧 Building model on CPU")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.cpu()

    ratio = {"Medium": 0.20, "Silver": 0.35, "Light": 0.40}.get(name, 0.0)

    if ratio > 0:
        dummy = torch.randn(1, 3, 224, 224)
        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy,
            importance=tp.importance.MagnitudeImportance(p=1),
            pruning_ratio=ratio,
            ignored_layers=[model.fc]
        )
        pruner.step()

    print("📦 Loading checkpoint")
    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict):
        state.pop("fc.weight", None)
        state.pop("fc.bias", None)
        model.load_state_dict(state, strict=False)
    else:
        model = state

    print(f"🚀 Moving model to {device}")
    try:
        model.to(device)
    except Exception as e:
        print("❌ GPU move failed → Using CPU")
        print(str(e))
        model.cpu()
        device = torch.device("cpu")

    return model

# ======================================================
# FREEZE / UNFREEZE
# ======================================================
def set_trainable_params(model, stage):
    for p in model.parameters():
        p.requires_grad = False

    layers = [model.fc, model.layer4]
    if stage == 2:
        layers.append(model.layer3)

    for m in layers:
        for p in m.parameters():
            p.requires_grad = True

# ======================================================
# TRAIN / EVAL
# ======================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        x, y_a, y_b, lam = mixup_data(x, y)
        out = model(x)
        loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (lam * (pred == y_a) + (1 - lam) * (pred == y_b)).sum().item()
        total += y.size(0)

    return correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ======================================================
# MAIN
# ======================================================
def main():
    device = get_safe_device()
    loaders = get_dataloaders(device)
    train_loader, val_loader, test_loader = loaders

    os.makedirs(SAVE_DIR, exist_ok=True)

    variants = {
        "Medium": MEDIUM_MODEL_PATH,
        "Silver": SILVER_MODEL_PATH,
        "Light": LIGHT_MODEL_PATH,
    }

    for name, path in variants.items():
        print(f"\n===== {name} =====")
        model = build_pruned_model(name, path, device)

        set_trainable_params(model, stage=1)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LR_FC, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best, patience = 0.0, 0

        for e in range(EPOCHS):
            if e == STAGE2_EPOCH:
                print("🔓 Stage 2: Unfreeze layer3")
                set_trainable_params(model, stage=2)

            train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            print(f"Epoch {e+1:02d} | Train {train_acc:.3f} | Val {val_acc:.3f}")

            if val_acc > best:
                best = val_acc
                patience = 0
                torch.save(model.state_dict(), f"{SAVE_DIR}/{name}_best.pth")
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print("⏹ Early stopping")
                    break

        model.load_state_dict(torch.load(f"{SAVE_DIR}/{name}_best.pth", map_location=device))
        test_acc = evaluate(model, test_loader, device)
        print(f"✅ Final Test Acc: {test_acc:.3f}")

if __name__ == "__main__":
    main()
