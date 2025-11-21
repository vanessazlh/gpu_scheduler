"""
GPU Scheduler Fairness Measurement - Optimized Version
Clean structure + essential statistical analysis
"""

import torch
import time
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------
# Simple GPU workload class
# -------------------------
class SimpleGPUWorkload:
    def __init__(self, memory_mb, duration_sec, workload_id):
        self.memory_mb = memory_mb
        self.duration_sec = duration_sec
        self.workload_id = workload_id
        self.tensor = None

    def allocate_memory(self):
        """Allocate GPU memory safely"""
        try:
            elements = (self.memory_mb * 1024 * 1024) // 4
            side = int(np.sqrt(elements))
            self.tensor = torch.randn((side, side), device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            return True
        except RuntimeError as e:
            print(f"[OOM] Job {self.workload_id} failed to allocate {self.memory_mb}MB")
            self.tensor = None
            return False

    def run(self):
        if self.tensor is None:
            return 0.0
        
        t_start = time.time()
        while time.time() - t_start < self.duration_sec:
            tmp = torch.sin(self.tensor * 2 + 1)
            torch.cuda.synchronize()
        
        return time.time() - t_start

    def free(self):
        if self.tensor is not None:
            del self.tensor
            self.tensor = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.synchronize()

# -------------------------
# Sequential execution (batch arrival)
# -------------------------
def run_sequential(workloads_config, scenario_name, trial_num):
    """All jobs arrive at t=0, executed in FIFO order"""
    results = []
    cumulative_time = 0.0
    
    for mem, dur, wid in workloads_config:
        job = SimpleGPUWorkload(mem, dur, wid)
        
        if not job.allocate_memory():
            results.append({
                'trial': trial_num,
                'workload_id': wid,
                'memory_mb': mem,
                'status': 'OOM_failure'
            })
            continue
        
        exec_time = job.run()
        
        results.append({
            'trial': trial_num,
            'workload_id': wid,
            'memory_mb': mem,
            'arrival_time': 0.0,
            'completion_time': cumulative_time + exec_time,
            'turnaround_time': cumulative_time + exec_time,
            'wait_time': cumulative_time,
            'status': 'success'
        })
        
        cumulative_time += exec_time
        job.free()
        time.sleep(0.1)  # Brief pause between jobs
    
    return results

# -------------------------
# Concurrent execution (staggered arrival)
# -------------------------
def run_concurrent(workloads_config, scenario_name, trial_num):
    """Jobs arrive at staggered times"""
    import threading
    results = []
    results_lock = threading.Lock()
    trial_start = time.time()
    
    def run_job(mem, dur, wid, offset):
        time.sleep(offset)
        arrival = time.time() - trial_start
        
        job = SimpleGPUWorkload(mem, dur, wid)
        if not job.allocate_memory():
            with results_lock:
                results.append({
                    'trial': trial_num,
                    'workload_id': wid,
                    'memory_mb': mem,
                    'arrival_time': arrival,
                    'status': 'OOM_failure'
                })
            return
        
        exec_time = job.run()
        completion = time.time() - trial_start
        
        with results_lock:
            results.append({
                'trial': trial_num,
                'workload_id': wid,
                'memory_mb': mem,
                'arrival_time': arrival,
                'completion_time': completion,
                'turnaround_time': completion - arrival,
                'status': 'success'
            })
        
        job.free()
    
    threads = []
    for i, (mem, dur, wid) in enumerate(workloads_config):
        t = threading.Thread(target=run_job, args=(mem, dur, wid, i * 0.5))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    return results

# -------------------------
# Run multiple trials
# -------------------------
def run_experiment(workloads_config, scenario_name, num_trials=12, concurrent=False):
    """Run experiment with multiple trials"""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name} ({'concurrent' if concurrent else 'sequential'})")
    print(f"Trials: {num_trials}")
    print(f"{'='*60}")
    
    all_results = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial+1}/{num_trials}...", end=' ')
        
        if concurrent:
            trial_results = run_concurrent(workloads_config, scenario_name, trial)
        else:
            trial_results = run_sequential(workloads_config, scenario_name, trial)
        
        all_results.extend(trial_results)
        print("✓")
        
        torch.cuda.empty_cache()
        time.sleep(0.5)
    
    return pd.DataFrame(all_results)

# -------------------------
# Jain's Fairness Index
# -------------------------
def jains_index(turnaround_times):
    if len(turnaround_times) == 0:
        return np.nan
    sum_x = np.sum(turnaround_times)
    sum_x2 = np.sum(turnaround_times ** 2)
    return (sum_x ** 2) / (len(turnaround_times) * sum_x2) if sum_x2 > 0 else np.nan

# -------------------------
# Statistical analysis
# -------------------------
def analyze_results(results_dict):
    """Perform statistical analysis"""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # 1. Fairness comparison
    print("\n1. FAIRNESS METRICS (Jain's Index)")
    print("-" * 80)
    fairness_data = {}
    
    for scenario, df in results_dict.items():
        df_ok = df[df['status'] == 'success']
        if len(df_ok) > 0:
            fairness = jains_index(df_ok['turnaround_time'].values)
            mean_tt = df_ok['turnaround_time'].mean()
            std_tt = df_ok['turnaround_time'].std()
            fairness_data[scenario] = fairness
            
            print(f"{scenario:40s}: {fairness:.4f} (mean={mean_tt:.3f}s, std={std_tt:.3f}s)")
    
    # 2. Sequential vs Concurrent comparison
    print("\n2. SEQUENTIAL vs CONCURRENT (T-Tests)")
    print("-" * 80)
    
    pairs = [
        ('homogeneous_sequential', 'homogeneous_concurrent'),
        ('heterogeneous_sequential', 'heterogeneous_concurrent')
    ]
    
    for seq_name, conc_name in pairs:
        if seq_name in results_dict and conc_name in results_dict:
            seq_df = results_dict[seq_name]
            conc_df = results_dict[conc_name]
            
            seq_ok = seq_df[seq_df['status'] == 'success']
            conc_ok = conc_df[conc_df['status'] == 'success']
            
            if len(seq_ok) > 0 and len(conc_ok) > 0:
                t_stat, p_val = stats.ttest_ind(
                    seq_ok['turnaround_time'],
                    conc_ok['turnaround_time']
                )
                
                print(f"\n{seq_name.split('_')[0].title()}:")
                print(f"  Sequential: {seq_ok['turnaround_time'].mean():.3f}s")
                print(f"  Concurrent: {conc_ok['turnaround_time'].mean():.3f}s")
                print(f"  T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
                print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'} (α=0.05)")
    
    # 3. Small_1 vs Small_2 analysis
    print("\n3. SMALL_1 vs SMALL_2 (Why does Small_2 take longer?)")
    print("-" * 80)
    
    for scenario in ['heterogeneous_sequential', 'heterogeneous_concurrent']:
        if scenario in results_dict:
            df = results_dict[scenario][results_dict[scenario]['status'] == 'success']
            
            s1 = df[df['workload_id'] == 'small_1']['turnaround_time']
            s2 = df[df['workload_id'] == 'small_2']['turnaround_time']
            
            if len(s1) > 0 and len(s2) > 0:
                t_stat, p_val = stats.ttest_ind(s1, s2)
                
                print(f"\n{scenario}:")
                print(f"  Small_1: {s1.mean():.3f}s (n={len(s1)})")
                print(f"  Small_2: {s2.mean():.3f}s (n={len(s2)})")
                print(f"  Difference: {s2.mean() - s1.mean():.3f}s")
                print(f"  T-stat: {t_stat:.4f}, p-value: {p_val:.4f}")
                print(f"  Significant: {'YES' if p_val < 0.05 else 'NO'}")
                
                if 'wait_time' in df.columns:
                    s1_wait = df[df['workload_id'] == 'small_1']['wait_time'].mean()
                    s2_wait = df[df['workload_id'] == 'small_2']['wait_time'].mean()
                    print(f"  → Small_2 waits {s2_wait - s1_wait:.3f}s longer in queue")
    
    return fairness_data

# -------------------------
# Visualization
# -------------------------
def create_plots(results_dict, timestamp):
    """Create fairness comparison bar chart"""
    fairness_values = []
    labels = []
    colors = []
    
    for scenario, df in results_dict.items():
        df_ok = df[df['status'] == 'success']
        if len(df_ok) > 0:
            fairness = jains_index(df_ok['turnaround_time'].values)
            fairness_values.append(fairness)
            labels.append(scenario.replace('_', ' ').title())
            colors.append('#3498db' if 'sequential' in scenario else '#e74c3c')
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(fairness_values)), fairness_values, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Jain's Fairness Index", fontsize=12, fontweight='bold')
    ax.set_title("GPU Scheduler Fairness Comparison (12 Trials)", 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, 
               alpha=0.7, label='Perfect Fairness')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, fairness_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#3498db', label='Sequential (Batch)'),
        Patch(facecolor='#e74c3c', label='Concurrent (Staggered)')
    ]
    ax.legend(handles=legend, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    filename = f'fairness_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {filename}")
    plt.show()

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        exit(1)
    
    # GPU info
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_mem_gb = gpu_props.total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {gpu_mem_gb:.2f} GB")
    
    # Safe memory settings
    safe_mem_per_job = min(int(gpu_mem_gb * 1024 * 0.3), 1024)
    print(f"Memory per job: {safe_mem_per_job}MB")
    
    # Define workloads
    homogeneous = [(safe_mem_per_job, 2, f'job_{i+1}') for i in range(4)]
    heterogeneous = [
        (safe_mem_per_job // 3, 1, 'small_1'),
        (safe_mem_per_job, 3, 'large_1'),
        (safe_mem_per_job // 3, 1, 'small_2'),
        (safe_mem_per_job, 3, 'large_2'),
    ]
    
    # Run experiments (12 trials each)
    results = {}
    results['homogeneous_sequential'] = run_experiment(
        homogeneous, 'homogeneous', num_trials=12, concurrent=False)
    
    results['heterogeneous_sequential'] = run_experiment(
        heterogeneous, 'heterogeneous', num_trials=12, concurrent=False)
    
    results['homogeneous_concurrent'] = run_experiment(
        homogeneous, 'homogeneous', num_trials=12, concurrent=True)
    
    results['heterogeneous_concurrent'] = run_experiment(
        heterogeneous, 'heterogeneous', num_trials=12, concurrent=True)
    
    # Save CSVs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    for scenario, df in results.items():
        filename = f'results_{scenario}_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"✓ {filename}")
    
    # Statistical analysis
    fairness_data = analyze_results(results)
    
    # Create visualization
    create_plots(results, timestamp)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)