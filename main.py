"""
MAIN ORCHESTRATOR: Carbon-Aware Stability-Constrained Model Switching

Pipeline:
  1. Load carbon intensity traces (real or synthetic)
  2. Initialize model zoo (Base, Medium, Silver, Light variants)
  3. Configure schedulers (proposed + baselines for comparison)
  4. Run simulation with dynamic model switching
  5. Evaluate and compare results
  6. Generate plots and reports

This is the entry point for the full system.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local imports
from src.models.model_zoo import ModelZoo
from src.scheduler.proposed import DeltaHysteresisScheduler
from src.scheduler.baselines import AlwaysBaselineScheduler, StaticScheduler
from src.carbon.loader import CarbonTraceLoader, CarbonSignal
from src.simulation.engine import InferenceSimulator
from src.evaluation.metrics import compute_accuracy_metrics, compute_stability_metrics, compare_schedulers


def setup_data(data_dir: str, batch_size: int = 128) -> DataLoader:
    """
    Setup CIFAR-10 test dataset.
    
    Args:
        data_dir: Path to CIFAR-10 data
        batch_size: Batch size for evaluation
        
    Returns:
        Test DataLoader
    """
    print("📂 Setting up dataset...")
    
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    
    test_set = datasets.ImageFolder(
        root=Path(data_dir) / "test" if Path(data_dir / "test").exists() else data_dir,
        transform=val_transform
    )
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"  ✅ Test set: {len(test_set)} images")
    return test_loader


def load_carbon_data(carbon_source: str, days: int = 14) -> CarbonSignal:
    """
    Load carbon intensity trace.
    
    Args:
        carbon_source: Path to CSV or 'synthetic'
        days: Number of days (for synthetic)
        
    Returns:
        CarbonSignal object
    """
    print("⚡ Loading carbon trace...")
    
    loader = CarbonTraceLoader()
    
    if carbon_source.lower() == 'synthetic':
        trace = loader.generate_synthetic(days=days, pattern='realistic')
        print(f"  ✅ Generated synthetic trace: {len(trace)} timesteps")
    else:
        trace = loader.load_csv(carbon_source)
        print(f"  ✅ Loaded real trace from: {carbon_source}")
    
    metadata = loader.get_metadata()
    print(f"  📊 Stats: mean={metadata['mean']:.0f}, "
          f"std={metadata['std']:.0f}, "
          f"range=[{metadata['min']:.0f}, {metadata['max']:.0f}] g/kWh")
    
    return CarbonSignal(trace)


def initialize_models(models_dir: str, device: str = "cuda") -> ModelZoo:
    """
    Initialize model zoo.
    
    Args:
        models_dir: Directory containing .pth files
        device: Torch device
        
    Returns:
        Initialized ModelZoo
    """
    print("🤖 Initializing model zoo...")
    
    zoo = ModelZoo(models_dir=models_dir, device=device)
    
    print(f"  ✅ Available models: {', '.join(zoo.registry.keys())}")
    print("\n" + zoo.compare_variants())
    
    return zoo


def run_simulation(
    scheduler,
    scheduler_name: str,
    model_zoo: ModelZoo,
    carbon_signal: CarbonSignal,
    test_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """
    Run inference simulation with given scheduler.
    
    Args:
        scheduler: Scheduler instance
        scheduler_name: Name for logging
        model_zoo: Initialized ModelZoo
        carbon_signal: CarbonSignal instance
        test_loader: Test DataLoader
        device: Torch device
        
    Returns:
        Results dictionary
    """
    print(f"\n🔄 SIMULATING: {scheduler_name}")
    print("-" * 60)
    
    simulator = InferenceSimulator(
        model_zoo=model_zoo,
        scheduler=scheduler,
        carbon_signal=carbon_signal,
        test_loader=test_loader,
        device=device
    )
    
    results = simulator.run(verbose=True)
    trace_data = simulator.get_trace_data()
    
    return {
        'name': scheduler_name,
        'results': results,
        'trace': trace_data,
        'scheduler': scheduler
    }


def main():
    """Main orchestration function."""
    
    parser = argparse.ArgumentParser(
        description="Carbon-Aware Stability-Constrained Model Switching"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/cifar10',
        help='Path to CIFAR-10 dataset'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Path to model weights directory'
    )
    parser.add_argument(
        '--carbon-source',
        type=str,
        default='synthetic',
        help='Carbon data source (CSV path or "synthetic")'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='Number of days for synthetic trace'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device (cuda or cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1043,
        help='Scheduler alpha threshold'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for plots and reports'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 80)
    print("🌍 CARBON-AWARE STABILITY-CONSTRAINED MODEL SWITCHING")
    print("=" * 80)
    print(f"📌 Device: {args.device}")
    print(f"📌 Data: {args.data_dir}")
    print(f"📌 Models: {args.models_dir}")
    print(f"📌 Carbon: {args.carbon_source}")
    print("=" * 80 + "\n")
    
    # ============================================================
    # SETUP
    # ============================================================
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # 1. Load data
    test_loader = setup_data(args.data_dir, batch_size=args.batch_size)
    
    # 2. Load carbon traces (one for each scheduler)
    carbon_signal_proposed = load_carbon_data(args.carbon_source, days=args.days)
    carbon_signal_baseline = load_carbon_data(args.carbon_source, days=args.days)
    carbon_signal_static = load_carbon_data(args.carbon_source, days=args.days)
    
    # 3. Initialize models
    model_zoo = initialize_models(args.models_dir, device=device)
    
    # ============================================================
    # CONFIGURE SCHEDULERS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURING SCHEDULERS")
    print("=" * 80)
    
    schedulers = [
        (
            DeltaHysteresisScheduler(alpha=args.alpha),
            f"Proposed (α={args.alpha})",
            carbon_signal_proposed
        ),
        (
            AlwaysBaselineScheduler(),
            "Baseline (Always Base)",
            carbon_signal_baseline
        ),
        (
            StaticScheduler(threshold=400.0, lower_model="Silver"),
            "Static (Fixed Threshold)",
            carbon_signal_static
        ),
    ]
    
    # ============================================================
    # RUN SIMULATIONS
    # ============================================================
    
    print("\n" + "=" * 80)
    print("🔬 RUNNING SIMULATIONS")
    print("=" * 80)
    
    all_results = []
    
    for scheduler, name, carbon_signal in schedulers:
        result = run_simulation(
            scheduler=scheduler,
            scheduler_name=name,
            model_zoo=model_zoo,
            carbon_signal=carbon_signal,
            test_loader=test_loader,
            device=device
        )
        all_results.append(result)
    
    # ============================================================
    # EVALUATE & COMPARE
    # ============================================================
    
    print("\n" + "=" * 80)
    print("📊 EVALUATION & COMPARISON")
    print("=" * 80)
    
    # Print comparison table
    scheduler_names = [r['name'] for r in all_results]
    result_dicts = [r['results'] for r in all_results]
    
    print("\n" + compare_schedulers(result_dicts, scheduler_names))
    
    # Detailed metrics for proposed scheduler
    proposed = all_results[0]
    print(f"\n✨ PROPOSED SCHEDULER DETAILED METRICS:")
    print(f"   Accuracy:     {proposed['results']['mean_accuracy']:.2f}% (±{proposed['results']['std_accuracy']:.2f}%)")
    print(f"   Stability:    {proposed['results']['switches_per_100_steps']:.2f} switches per 100 steps")
    print(f"   Carbon Cost:  {proposed['results']['total_carbon_cost']:.2f} units")
    print(f"   Model Usage:  {proposed['results']['model_usage']}")
    print(f"   Latency:      {proposed['results']['mean_latency_ms']:.2f} ms/inference")
    
    # ============================================================
    # GENERATE PLOTS
    # ============================================================
    
    print(f"\n📈 Generating plots...")
    
    # Plot 1: Carbon intensity over time with model selections
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    proposed_trace = proposed['trace']
    timesteps = proposed_trace['timesteps']
    carbon = proposed_trace['carbon_intensities']
    models = proposed_trace['model_indices']
    
    ax1.plot(timesteps, carbon, 'b-', linewidth=2, label='Carbon Intensity')
    ax1.fill_between(timesteps, carbon, alpha=0.3)
    ax1.set_ylabel('Carbon Intensity (g/kWh)', fontsize=12)
    ax1.set_title('Carbon Intensity Trace', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.step(timesteps, models, where='post', linewidth=2, color='red')
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Model Index', fontsize=12)
    ax2.set_title(f'Proposed Scheduler: Model Selection Over Time (Total Switches: {proposed["results"]["total_switches"]})', 
                  fontsize=14, fontweight='bold')
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Base', 'Medium', 'Silver', 'Light'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(args.output_dir) / 'model_selection_trace.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Scheduler comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    scheduler_names_short = [r['name'] for r in all_results]
    accuracies = [r['results']['mean_accuracy'] for r in all_results]
    carbon_costs = [r['results']['total_carbon_cost'] for r in all_results]
    
    axes[0].bar(scheduler_names_short, accuracies, color=['green', 'gray', 'orange'])
    axes[0].set_ylabel('Mean Accuracy (%)', fontsize=11)
    axes[0].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim([80, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(scheduler_names_short, carbon_costs, color=['green', 'gray', 'orange'])
    axes[1].set_ylabel('Total Carbon Cost', fontsize=11)
    axes[1].set_title('Carbon Cost Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = Path(args.output_dir) / 'scheduler_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"📂 Results saved to: {args.output_dir}/")
    print("\nKey Insight:")
    print(f"  Proposed scheduler achieves {proposed['results']['mean_accuracy']:.1f}% accuracy")
    print(f"  with {proposed['results']['total_switches']} model switches")
    print(f"  (vs {all_results[1]['results']['total_switches']} for baseline)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
