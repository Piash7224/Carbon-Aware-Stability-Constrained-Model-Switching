import os, time, torch, collections
import torch.nn as nn
import torch_pruning as tp
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Subset

# =====================================================
# CONFIG
# =====================================================
DATA_DIR   = "/kaggle/input/cifar10-pngs-in-folders/cifar10"
MODEL_DIR  = "/kaggle/input/cifar10-pruned/models"
CAISO_PATH = "/kaggle/input/carbon-grid/csv/CAISO_NORTH_WattTime.csv"
PJM_PATH   = "/kaggle/input/carbon-grid/csv/PJM_WESTERN_KY_WattTime.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

# Relative energy proxy (GMAC-based)
MODEL_ENERGY_J = {
    "Base":   1.00,
    "Medium": 0.64,
    "Silver": 0.43,
    "Light":  0.36
}

# =====================================================
# DATASET
# =====================================================
def get_test_loader():
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])
    ds = datasets.ImageFolder(os.path.join(DATA_DIR,"test"), tfm)

    # USE FULL TEST SET (10k)
    idx = torch.arange(len(ds))
    return DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)

# =====================================================
# MODEL LOADING + PRUNING (FIXED)
# =====================================================
def apply_pruning(model, name):
    # 1. Define Ratios
    target_ratio = 0.0
    if "Medium" in name: target_ratio = 0.20
    elif "Silver" in name: target_ratio = 0.35
    elif "Light" in name: target_ratio = 0.40

    # 2. Apply Pruning only if ratio > 0 (Skip for Base)
    if target_ratio > 0:
        # Dummy input for graph tracing
        dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
        model.to(DEVICE) # Ensure model is on device before pruning trace
        
        imp = tp.importance.MagnitudeImportance(p=1)

        # Ignore the classification layer (fc)
        ignored = []
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.out_features == 10:
                ignored.append(m)

        pruner = tp.pruner.MagnitudePruner(
            model,
            dummy,
            importance=imp,
            pruning_ratio=target_ratio,
            iterative_steps=1,
            ignored_layers=ignored
        )
        pruner.step()
        
    return model

def load_pruned_model(name):
    # A. Init Standard ResNet50
    model = resnet50(weights=None)
    
    # B. Adjust for CIFAR-10 (10 classes)
    # Must be done BEFORE pruning to establish correct graph connectivity
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # C. Apply Pruning (Shrinks the architecture)
    model = apply_pruning(model, name)
    
    # D. Load Weights
    weight_path = os.path.join(MODEL_DIR, f"{name}_best.pth")
    
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {name} successfully.")
    else:
        print(f"WARNING: Could not find {weight_path}. Returning random init model.")
        
    model.to(DEVICE)
    model.eval()
    return model

# =====================================================
# SCHEDULERS
# =====================================================
class Proposed:
    def __init__(self, alpha=0.1043):
        self.h = collections.deque(maxlen=5)
        self.cur = "Base"
        self.a = alpha
        self.name = "Proposed"

    def decide(self, c):
        self.h.append(c)
        if len(self.h)<3: return self.cur
        mu = np.mean(self.h)
        d = (c-mu)/mu
        if d > 3*self.a: self.cur="Light"
        elif d > self.a: self.cur="Silver"
        elif d < -self.a: self.cur="Base"
        else:
            if self.cur=="Light": self.cur="Silver"
            elif self.cur=="Base": self.cur="Medium"
        return self.cur

class Static:
    def __init__(self, t):
        self.lo,self.hi = np.percentile(t,[30,70])
        self.name="Static"
    def decide(self,c):
        return "Base" if c<self.lo else "Silver" if c<self.hi else "Light"

class NoHyst:
    def __init__(self):
        self.h=collections.deque(maxlen=5)
        self.cur="Base"
        self.name="NoHyst"
    def decide(self,c):
        self.h.append(c)
        if len(self.h)<3: return self.cur
        mu=np.mean(self.h); d=(c-mu)/mu
        self.cur = "Light" if d>0.1 else "Silver" if d>0 else "Base"
        return self.cur

class Binary:
    def __init__(self):
        self.h=collections.deque(maxlen=5)
        self.cur="Base"
        self.name="Binary"
    def decide(self,c):
        self.h.append(c)
        if len(self.h)<3: return self.cur
        mu=np.mean(self.h); d=(c-mu)/mu
        self.cur = "Light" if d>0.05 else "Base"
        return self.cur

class AlwaysBase:
    def __init__(self):
        self.name="Baseline"
    def decide(self,c):
        return "Base"

# =====================================================
# SIMULATION + CO2
# =====================================================
def run_sim(trace, sched, models, loader):
    # Reset scheduler
    if hasattr(sched, "cur"): sched.cur="Base"
    if hasattr(sched,'h'): sched.h.clear()

    sel, sw, lat, cor, tot = [], 0, 0, 0, 0
    prev="Base"

    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            c = trace[i % len(trace)]
            mname = sched.decide(c)
            sel.append(mname)
            if mname!=prev: sw+=1; prev=mname
            m = models[mname]
            x,y=x.to(DEVICE),y.to(DEVICE)

            # GPU Latency correction
            t0=time.time()
            o=m(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            lat+=(time.time()-t0)*1000

            cor+=(o.argmax(1)==y).sum().item()
            tot+=1

    # CO2 (relative)
    co2=0
    for i,mn in enumerate(sel):
        co2 += trace[i % len(trace)] * MODEL_ENERGY_J[mn]

    return cor/tot, sw, lat/tot, co2

# =====================================================
# MAIN
# =====================================================
# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    # 1. Load Models
    print("Loading Models...")
    models = {k: load_pruned_model(k) for k in ["Base", "Medium", "Silver", "Light"]}
    
    loader = get_test_loader()

    # 2. Robust CSV Loading
    def get_trace(path):
        df = pd.read_csv(path)
        if "MARGINAL_EMISSIONS_KG_PER_MWH" in df.columns:
            return df["MARGINAL_EMISSIONS_KG_PER_MWH"].values
        elif "co2" in df.columns:
            return df["co2"].values
        return df.iloc[:, 0].values

    caiso = get_trace(CAISO_PATH)[:336]
    pjm   = get_trace(PJM_PATH)[:336]

    for region, trace in [("CAISO", caiso), ("PJM", pjm)]:
        print(f"\n=== {region} ===")

        # Baseline (always Base)
        base_acc, base_sw, base_lat, base_co2 = run_sim(trace, AlwaysBase(), models, loader)

        print(f"Baseline  | Acc {base_acc:.4f} | Sw {base_sw:4d} | "
              f"Lat {base_lat:.2f}ms | CO2 {base_co2:.3f} | Eff 0.000")

        for S in [Proposed(), Static(trace), NoHyst(), Binary()]:
            acc, sw, lat, co2 = run_sim(trace, S, models, loader)
            
            saved = base_co2 - co2
            eff = saved / sw if sw > 0 else 0.0
            
            print(f"{S.name:8s} | Acc {acc:.4f} | Sw {sw:4d} | "
                  f"Lat {lat:.2f}ms | CO2 {co2:.3f} | Eff {eff:.3f}")

