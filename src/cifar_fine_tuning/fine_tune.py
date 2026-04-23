import os
import torch
import torch.nn as nn
import torch.optim as optim
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
BASE_MODEL_PATH   = "/kaggle/input/family-of-model/models/resnet50_base_best.pth"
MEDIUM_MODEL_PATH = "/kaggle/input/pruned-models/pruned/resnet50_med_best.pth"
SILVER_MODEL_PATH = "/kaggle/input/pruned-models/pruned/resnet50_silver_final_best.pth"
LIGHT_MODEL_PATH  = "/kaggle/input/pruned-models/pruned/resnet50_light_RESCUED.pth"

SAVE_DIR = "/kaggle/working/fine_tuned"

# ======================================================
# HYPERPARAMETERS
# ======================================================
NUM_CLASSES = 10
BATCH_SIZE = 128
LR_FC = 3e-3
LR_LAYER4 = 1e-3
LR_LAYER3 = 5e-4
EPOCHS = 25
STAGE2_EPOCH = 8
EARLY_STOPPING_PATIENCE = 5
SEED = 42
MIXUP_ALPHA = 0.4

torch.manual_seed(SEED)

# ======================================================
# DATASET + AUGMENTATION
# ======================================================
def get_dataloaders():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
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

    full_train = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=train_tf)
    train_len = int(0.9 * len(full_train))
    val_len = len(full_train) - train_len
    train_set, val_set = random_split(full_train, [train_len, val_len],
                                      generator=torch.Generator().manual_seed(SEED))
    # Apply clean transform for validation
    val_set.dataset.transform = val_tf

    test_set = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=val_tf)

    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
        DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
    )

# ======================================================
# MIXUP FUNCTIONS
# ======================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================================================
# BUILD MODEL
# ======================================================
def build_model(path, device):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    state = torch.load(path, map_location=device)
    # Remove fc weights if they exist
    if 'fc.weight' in state and 'fc.bias' in state:
        del state['fc.weight']
        del state['fc.bias']

    model.load_state_dict(state, strict=False)
    return model

# ======================================================
# FREEZE / UNFREEZE
# ======================================================
def set_trainable_params(model, stage=1):
    if stage == 1:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif stage == 2:
        for param in model.layer3.parameters():
            param.requires_grad = True

# ======================================================
# TRAIN / EVAL
# ======================================================
def train_one_epoch(model, loader, optimizer, criterion, device, mixup=True):
    model.train()
    correct, total = 0, 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if mixup:
            x, y_a, y_b, lam = mixup_data(x, y)
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            pred = out.argmax(1)
            correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
        else:
            out = model(x)
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()

        loss.backward()
        optimizer.step()
        total += y.size(0)
    return correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ======================================================
# MAIN
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()
    os.makedirs(SAVE_DIR, exist_ok=True)

    variants = {
        "Base": BASE_MODEL_PATH,
        "Medium": MEDIUM_MODEL_PATH,
        "Silver": SILVER_MODEL_PATH,
        "Light": LIGHT_MODEL_PATH,
    }

    for name, path in variants.items():
        print(f"\n===== {name} =====")
        model = build_model(path, device)
        set_trainable_params(model, stage=1)

        optimizer = optim.SGD([
            {"params": model.fc.parameters(), "lr": LR_FC},
            {"params": model.layer4.parameters(), "lr": LR_LAYER4},
            {"params": model.layer3.parameters(), "lr": LR_LAYER3},
        ], momentum=0.9, weight_decay=5e-4)

        total_steps = len(train_loader) * EPOCHS
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

        criterion = nn.CrossEntropyLoss()
        best_val = 0.0
        patience = 0

        for e in range(EPOCHS):
            if e == STAGE2_EPOCH:
                set_trainable_params(model, stage=2)
                print("Stage 2: Unfreezing layer3 for partial fine-tuning")

            train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup=True)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()

            print(f"Epoch {e+1:02d}: Train {train_acc:.3f} | Val {val_acc:.3f}")

            if val_acc > best_val:
                best_val = val_acc
                patience = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_best.pth"))
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f"{name}_best.pth")))
        test_acc = evaluate(model, test_loader, device)
        print(f"Final Test Acc: {test_acc:.3f}")

if __name__ == "__main__":
    main()
