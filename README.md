You’re close—but right now your README still mixes **demo-style claims + paper results**, which is risky. I’ve rewritten it fully so it is:

* ✅ 100% consistent with your paper
* ✅ Reviewer-safe (no exaggerated claims)
* ✅ Clean, professional research repo standard
* ✅ Mentions **ICEFront submission (under review)** properly

---

# ✅ **FINAL CLEAN README (USE THIS)**

```markdown
# 🌍 Carbon-Aware Stability-Constrained Model Switching

**Research Framework for Dynamic Inference with Environmental Constraints**

This repository contains the implementation of a **carbon-aware inference framework** that dynamically adapts model complexity based on real-time grid carbon intensity, while ensuring **stability and reliability in deployment**.

📌 **Paper Status:** Under review at ICEFront 2026

---

## 🎯 The Problem

Modern AI systems increasingly contribute to global carbon emissions, especially during large-scale inference. While grid carbon intensity fluctuates significantly over time, most deployed ML systems remain **carbon-blind**.

**Key Challenge:**  
How can we adapt inference dynamically to reduce emissions **without causing unstable model switching or degrading performance**?

---

## ✨ Our Contribution

We propose a **Carbon-Aware Stability-Constrained Model Switching Framework** featuring a **Delta-Hysteresis Scheduler**:

### 1. Carbon-Aware Decision Signal ⚡
- Uses real-time grid carbon intensity: \( C_{grid}(t) \)
- Computes deviation from moving average:
  
  \[
  \delta(t) = \frac{C_{grid}(t) - \mu(t)}{\mu(t)}
  \]

- Captures **relative environmental change**, not absolute thresholds

---

### 2. Stability-Constrained Scheduling 🎛️
- Multi-threshold decision logic using \( \alpha \) and \( 3\alpha \)
- Prevents oscillation via **hysteresis-based transitions**
- Ensures **stable adaptation under volatile carbon signals**

---

### 3. Pruning-Based Model Zoo 🤖
Four ResNet-50 variants providing controlled efficiency–accuracy trade-offs:

| Model   | Pruning | GMACs | Relative Energy |
|--------|--------|-------|----------------|
| Base   | 0%     | 4.12  | 1.00 |
| Medium | ~20%   | 2.65  | 0.64 |
| Silver | ~35%   | 1.76  | 0.43 |
| Light  | ~40%   | 1.50  | 0.36 |

---

### 4. Key Results 📊

From trace-driven evaluation (CAISO & PJM):

- **54–56% reduction** in inference-phase CO₂ emissions  
- **<4% absolute accuracy drop** vs baseline  
- **66–95% fewer switches** vs unstable schedulers  
- Robust across **volatile (CAISO)** and **stable (PJM)** grids  

---

## 🏗️ System Overview

```

Carbon Trace → Scheduler → Model Selection → Inference → Metrics

```

Core components:

- **Carbon Signal Module** → Provides \( C_{grid}(t) \)
- **Delta-Hysteresis Scheduler** → Selects model variant
- **Model Zoo** → Pruned ResNet-50 variants
- **Simulation Engine** → Evaluates performance & emissions

---

## 📁 Project Structure

```

Carbon-Aware-Stability-Constrained-Model-Switching/
│
├── main.py                 # Entry point (simulation pipeline)
│
├── src/
│   ├── models/             # Model definitions & loading
│   ├── scheduler/          # Proposed + baseline schedulers
│   ├── carbon/             # Carbon trace handling
│   ├── simulation/         # Inference simulation engine
│   ├── evaluation/         # Metrics computation
│   └── data/               # Dataset utilities
│
├── models/                 # (Optional) pretrained weights (not included)
├── results/                # Output plots & logs
│
├── requirements.txt
└── README.md

````

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Piash7224/DS-GCA-Multimodal-Survival-Prediction.git
cd Carbon-Aware-Stability-Constrained-Model-Switching
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Simulation

```bash
python main.py
```

---

## ⚙️ Experimental Setup

* Dataset: **CIFAR-10**
* Model: **Pruned ResNet-50 variants**
* Carbon Data:

  * CAISO (solar-heavy, volatile)
  * PJM (fossil-heavy, stable)
* Evaluation:

  * Accuracy
  * Model switching frequency
  * CO₂ emissions
  * Stability metric (GEM)

---

## 📊 Expected Results (Reproducible)

```
Proposed (Delta-Hysteresis):
  Accuracy:     ~69.5–69.8%
  CO₂ Saved:    ~54–56%
  Switches:     151–894

Baseline:
  Accuracy:     ~73.66%
  CO₂ Saved:    0%
  Switches:     0

Static / No-Hysteresis:
  Higher switching (2k–3k)
  Lower stability
  Reduced efficiency
```

---

## 🔬 Key Features

* Carbon-aware inference scheduling
* Stability-constrained switching (hysteresis)
* Structured pruning for efficiency scaling
* Trace-driven evaluation with real-world carbon data

---

## ⚠️ Important Notes

* Pretrained `.pth` models are **not included** (size constraints)
* Users should:

  * Train models OR
  * Provide their own checkpoints
* Carbon traces must be placed manually if not using synthetic data

---



## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mohammad Mahmud Hasan**
📧 [piashmahmud204@gmail.com]

---

## 🙏 Acknowledgments

* PyTorch & TorchVision
* torch-pruning library
* WattTime carbon datasets

