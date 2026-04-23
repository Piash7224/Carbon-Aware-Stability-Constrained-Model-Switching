import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import torch
import torchvision.models as models
import os
import seaborn as sns

# ==========================================
# 1. VISUAL CONFIGURATION (UPDATED FOR PRINT)
# ==========================================
np.random.seed(42) 
# INCREASED font_scale from 1.5 to 2.0 for better print visibility
sns.set_context("paper", font_scale=2.0) 
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'lines.linewidth': 4,       # Thicker lines
    'figure.dpi': 300,
    'axes.labelsize': 18,       # Explicit axis label size
    'axes.titlesize': 20,       # Explicit title size
    'xtick.labelsize': 16,      # Explicit tick label size
    'ytick.labelsize': 16,
    'legend.fontsize': 16       # Explicit legend size
})

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
SIMULATION_DAYS = 14
STEPS_PER_DAY = 96  # 15-min intervals
TOTAL_STEPS = SIMULATION_DAYS * STEPS_PER_DAY

BASE_DIR = '/kaggle/input/family-of-model/models/' 
PATH_CAISO = '/kaggle/input/dataset/csv/CAISO_NORTH_WattTime.csv'
PATH_PJM   = '/kaggle/input/dataset/csv/PJM_WESTERN_KY_WattTime.csv'

# Mocking model loads for simulation logic (paths strictly preserved)
MODEL_COSTS = {'Base (R50)': 1.0, 'Medium (Pr)': 0.8, 'Silver (Pr)': 0.65, 'Light (Pr)': 0.6}

# ==========================================
# 3. 14-DAY DATA GENERATOR
# ==========================================
def generate_14_day_trace():
    print(f"--- Step 1: Generating {SIMULATION_DAYS}-Day Simulation Data ---")
    
    # Load Real Data Patterns
    patterns = []
    for path, name in [(PATH_CAISO, "CAISO"), (PATH_PJM, "PJM")]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                best_col = None
                max_valid = 0
                for col in df.columns:
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() > max_valid:
                        max_valid = numeric.notna().sum()
                        best_col = numeric
                
                if best_col is not None:
                    data = best_col.dropna().values
                    data = np.interp(data, (data.min(), data.max()), (200, 800))
                    seed_day = data[:96] if len(data) >= 96 else np.resize(data, 96)
                    patterns.append(seed_day)
                    print(f" -> Loaded {name} Seed Pattern")
            except:
                pass
    
    # Synthetic Fallback
    if not patterns:
        print(" -> No CSVs found. Using Synthetic Seed.")
        x = np.linspace(0, 24, 96)
        seed = 400 + 150 * np.sin((x-6)/3.8) 
        patterns.append(seed)

    # TILE AND NOISE
    full_trace = []
    for day in range(SIMULATION_DAYS):
        base_pattern = patterns[day % len(patterns)]
        noise = np.random.normal(0, 30, 96) 
        daily_trace = base_pattern + noise
        if np.random.rand() > 0.7: 
            spike_start = np.random.randint(60, 80)
            # Clip index to prevent out of bounds
            end_idx = min(spike_start+10, 96)
            daily_trace[spike_start:end_idx] += 200
            
        full_trace.extend(daily_trace)
        
    full_trace = np.array(full_trace)
    full_trace = np.clip(full_trace, 150, 950) 
    return full_trace

carbon_trace = generate_14_day_trace()

# ==========================================
# 4. SCHEDULER LOGIC
# ==========================================
class ProposedScheduler:
    def __init__(self, alpha):
        self.history = collections.deque(maxlen=5) 
        self.current = 'Base (R50)'
        self.alpha = alpha
        self.spike = alpha * 3.0 
        
    def decide(self, co2):
        self.history.append(co2)
        thresh = np.mean(self.history)
        if thresh == 0: return self.current
        
        delta = (co2 - thresh) / thresh
        
        if delta > self.spike:
            self.current = 'Light (Pr)'
        elif (delta > self.alpha) and (delta <= self.spike):
            self.current = 'Silver (Pr)'
        elif delta < (-self.alpha):
            self.current = 'Base (R50)'
        else:
            if self.current == 'Light (Pr)': self.current = 'Silver (Pr)'
            elif self.current == 'Base (R50)': self.current = 'Medium (Pr)'
            
        return self.current

# ==========================================
# 5. SENSITIVITY ANALYSIS
# ==========================================
print("\n--- Step 2: Running 14-Day Sensitivity Sweep ---")
alphas = np.linspace(0.01, 0.15, 50) 
results = []
baseline_carbon = np.sum(carbon_trace * 1.0) 

for a in alphas:
    sched = ProposedScheduler(alpha=a)
    switches = 0
    carbon_sum = 0
    prev = 'Base (R50)'
    
    for val in carbon_trace:
        model = sched.decide(val)
        carbon_sum += (val * MODEL_COSTS[model])
        if model != prev:
            switches += 1
            prev = model
            
    savings = (1 - (carbon_sum / baseline_carbon)) * 100
    switches_per_day = switches / SIMULATION_DAYS
    stability_score = max(0, 100 - (switches_per_day * 4.0))
    score = (savings * 0.6) + (stability_score * 0.4)
    
    results.append({
        'Alpha': a * 100,
        'Stability': stability_score,
        'Savings': savings,
        'Score': score,
        'SwitchesDay': switches_per_day
    })

df = pd.DataFrame(results)
best_row = df.loc[df['Score'].idxmax()]
optimal_alpha = best_row['Alpha']

print(f"\n>>> 14-DAY RESULT: Optimal Alpha = {optimal_alpha:.2f}%")

# ==========================================
# 6. PROFESSIONAL DIAGRAM (UPDATED)
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 8)) # Increased figure size slightly

# Green Line: Stability
color_stab = '#2E7D32' 
ax1.plot(df['Alpha'], df['Stability'], color=color_stab, linestyle='--', label='System Stability', alpha=0.9)
ax1.set_xlabel(r'Base Threshold ($\alpha$ %)', fontweight='bold', fontsize=18)
ax1.set_ylabel('Stability Index (0-100)', color=color_stab, fontweight='bold', fontsize=18)
ax1.tick_params(axis='y', labelcolor=color_stab, labelsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.set_ylim(0, 105)

# Blue Line: Efficiency
ax2 = ax1.twinx()
color_eff = '#1565C0' 
ax2.plot(df['Alpha'], df['Savings'], color=color_eff, label='Carbon Efficiency', linewidth=4)
ax2.set_ylabel('Carbon Savings (%)', color=color_eff, fontweight='bold', fontsize=18)
ax2.tick_params(axis='y', labelcolor=color_eff, labelsize=16)

# Mark Optimal
plt.axvline(optimal_alpha, color='#37474F', linestyle='-', linewidth=2, alpha=0.3)
plt.scatter([optimal_alpha], [best_row['Savings']], color='#D32F2F', s=200, zorder=10, label='Optimal Point') # Larger dot

# Annotation
text = f"Optimal $\\alpha$={optimal_alpha:.1f}%\n(14-Day Mean)"
bbox = dict(boxstyle="round,pad=0.5", fc="white", ec="#37474F", alpha=0.9)
# Adjusted annotation position and font size
ax2.text(optimal_alpha + 0.5, best_row['Savings'] + 1, text, fontsize=14, bbox=bbox)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=16, frameon=True)

# REMOVED TITLE AS REQUESTED
# plt.title(f'Sensitivity Analysis: {SIMULATION_DAYS}-Day Robustness Test', pad=20, fontweight='bold', fontsize=22)

plt.tight_layout()
plt.savefig('sensitivity_analysis_print.pdf')
plt.show()