"""
GPU Scheduler Measurement Script
Measures actual GPU workload behavior for scheduler analysis
Run in Google Colab or any environment with GPU access
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class SimpleGPUWorkload:
    """A simple GPU workload that uses specified memory and compute time"""
    def __init__(self, memory_mb, duration_sec, workload_id):
        self.memory_mb = memory_mb
        self.duration_sec = duration_sec
        self.workload_id = workload_id
        self.tensor = None
        
    def allocate_memory(self):
        """Allocate GPU memory"""
        if torch.cuda.is_available():
            # Allocate tensor to use approximately memory_mb of GPU memory
            # Each float32 takes 4 bytes
            elements = (self.memory_mb * 1024 * 1024) // 4
            # Use 2D tensor for better alignment and more accurate memory allocation
            side = int(np.sqrt(elements))
            self.tensor = torch.randn((side, side), device='cuda', dtype=torch.float32)
            torch.cuda.synchronize()
            
            # Verify actual memory allocated
            actual_memory = self.tensor.element_size() * self.tensor.nelement() / (1024 * 1024)
            if abs(actual_memory - self.memory_mb) > self.memory_mb * 0.1:  # More than 10% off
                print(f"Warning: Requested {self.memory_mb}MB, allocated {actual_memory:.1f}MB")
    
    def run(self):
        """Run compute workload for specified duration"""
        if self.tensor is None:
            self.allocate_memory()
        
        start_time = time.time()
        
        # Based on benchmarking: ~0.028s per iteration for typical tensors
        # Do iterations in batches of 10 to reduce sync overhead
        batch_size = 10
        estimated_time_per_batch = 0.28  # 10 iterations × 0.028s
        
        while (time.time() - start_time) < self.duration_sec:
            # Perform a batch of iterations
            for _ in range(batch_size):
                temp = self.tensor * 2.0 + 1.0
                temp = torch.sin(temp)
            torch.cuda.synchronize()
            
            # Check if we're close to target time
            remaining = self.duration_sec - (time.time() - start_time)
            if remaining < estimated_time_per_batch:
                # Do final iterations without full batch
                final_iters = max(1, int(remaining / 0.028))
                for _ in range(final_iters):
                    temp = self.tensor * 2.0 + 1.0
                    temp = torch.sin(temp)
                torch.cuda.synchronize()
                break
    
    def free_memory(self):
        """Free GPU memory"""
        if self.tensor is not None:
            del self.tensor
            self.tensor = None
            torch.cuda.empty_cache()
            # Force garbage collection to ensure cleanup
            import gc
            gc.collect()
            torch.cuda.synchronize()

def measure_single_workload(memory_mb, duration_sec, workload_id):
    """Measure a single workload execution"""
    workload = SimpleGPUWorkload(memory_mb, duration_sec, workload_id)
    
    # Measure memory allocation time
    alloc_start = time.time()
    workload.allocate_memory()
    alloc_time = time.time() - alloc_start
    
    # Measure execution time
    exec_start = time.time()
    workload.run()
    exec_time = time.time() - exec_start
    
    # Measure deallocation time
    free_start = time.time()
    workload.free_memory()
    free_time = time.time() - free_start
    
    return {
        'workload_id': workload_id,
        'memory_mb': memory_mb,
        'target_duration_sec': duration_sec,
        'alloc_time_sec': alloc_time,
        'exec_time_sec': exec_time,
        'free_time_sec': free_time,
        'total_time_sec': alloc_time + exec_time + free_time
    }

def measure_sequential_workloads(workloads_config, trial_num):
    """
    Run workloads sequentially with batch arrival (all arrive at t=0)
    This simulates traditional FIFO scheduling
    """
    results = []
    
    # BATCH ARRIVAL: All jobs conceptually arrive at time 0
    arrival_time = 0.0
    
    # Track cumulative completion time
    cumulative_time = 0.0
    
    for i, (memory_mb, duration_sec, workload_id) in enumerate(workloads_config):
        # Execute the workload
        exec_start = time.time()
        result = measure_single_workload(memory_mb, duration_sec, workload_id)
        exec_end = time.time()
        
        # Actual execution time
        actual_exec = exec_end - exec_start
        
        # Calculate scheduling metrics
        result['trial'] = trial_num
        result['arrival_time'] = arrival_time  # All jobs arrive at 0
        result['execution_start'] = cumulative_time  # When it starts in the queue
        result['completion_time'] = cumulative_time + actual_exec  # When it finishes
        result['turnaround_time'] = result['completion_time'] - arrival_time  # Real turnaround!
        result['wait_time'] = cumulative_time - arrival_time  # Time waiting in queue
        
        results.append(result)
        
        # Update for next job
        cumulative_time += actual_exec
    
    return results

def measure_concurrent_workloads(workloads_config, trial_num):
    """
    Run workloads with staggered arrivals (concurrent submission)
    This shows how GPU handles multiple requests with overlapping submission
    """
    import threading
    results = []
    results_lock = threading.Lock()
    trial_start = time.time()
    
    def run_workload(memory_mb, duration_sec, workload_id, submission_offset):
        try:
            # Stagger submissions - jobs arrive at different times
            time.sleep(submission_offset)
            arrival_time = time.time() - trial_start
            
            # Execute the workload
            result = measure_single_workload(memory_mb, duration_sec, workload_id)
            completion_time = time.time() - trial_start
            
            result['trial'] = trial_num
            result['arrival_time'] = arrival_time
            result['completion_time'] = completion_time
            result['turnaround_time'] = completion_time - arrival_time
            result['status'] = 'success'
            
            with results_lock:
                results.append(result)
                
        except torch.cuda.OutOfMemoryError as e:
            # Record OOM failures
            with results_lock:
                results.append({
                    'workload_id': workload_id,
                    'memory_mb': memory_mb,
                    'target_duration_sec': duration_sec,
                    'trial': trial_num,
                    'arrival_time': time.time() - trial_start,
                    'status': 'OOM_failure',
                    'error': str(e)
                })
            torch.cuda.empty_cache()
    
    threads = []
    for i, (memory_mb, duration_sec, workload_id) in enumerate(workloads_config):
        submission_offset = i * 0.5  # Submit every 0.5 seconds
        thread = threading.Thread(
            target=run_workload,
            args=(memory_mb, duration_sec, workload_id, submission_offset)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return results

def calculate_fairness_metrics(results_df):
    """Calculate Jain's Fairness Index"""
    turnaround_times = results_df['turnaround_time'].values
    n = len(turnaround_times)
    
    if n == 0:
        return 0
    
    # Jain's Fairness Index: (sum of x_i)^2 / (n * sum of x_i^2)
    sum_x = np.sum(turnaround_times)
    sum_x_squared = np.sum(turnaround_times ** 2)
    
    jains_index = (sum_x ** 2) / (n * sum_x_squared) if sum_x_squared > 0 else 0
    
    return jains_index

def run_experiment(scenario_name, workloads_config, num_trials=3, concurrent=False):
    """Run a full experiment with multiple trials"""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {scenario_name}")
    print(f"Concurrent: {concurrent}, Trials: {num_trials}")
    print(f"{'='*60}")
    
    all_results = []
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...")
        
        if concurrent:
            trial_results = measure_concurrent_workloads(workloads_config, trial)
        else:
            trial_results = measure_sequential_workloads(workloads_config, trial)
        
        all_results.extend(trial_results)
        
        # Clear GPU memory between trials
        torch.cuda.empty_cache()
        time.sleep(1)
    
    df = pd.DataFrame(all_results)
    
    # Calculate metrics
    print(f"\nResults Summary:")
    print(f"Mean Turnaround Time: {df['turnaround_time'].mean():.3f} sec")
    print(f"Std Turnaround Time: {df['turnaround_time'].std():.3f} sec")
    print(f"Jain's Fairness Index: {calculate_fairness_metrics(df):.3f}")
    
    return df

# Main Experiment Runner
if __name__ == "__main__":
    # Check GPU availability and memory
    if not torch.cuda.is_available():
        print("ERROR: No GPU available!")
        exit(1)
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / 1e9
    
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
    
    # Adjust workload sizes based on GPU memory
    safe_memory_per_job = int((gpu_memory_gb * 0.5 * 1024) / 4)
    
    print(f"Safe memory per concurrent job: {safe_memory_per_job}MB")
    
    # Define workload scenarios
    homogeneous = [
        (min(1024, safe_memory_per_job), 2, 'job_1'),
        (min(1024, safe_memory_per_job), 2, 'job_2'),
        (min(1024, safe_memory_per_job), 2, 'job_3'),
        (min(1024, safe_memory_per_job), 2, 'job_4'),
    ]
    
    small_size = min(512, safe_memory_per_job // 3)
    large_size = min(1536, safe_memory_per_job)
    heterogeneous = [
        (small_size, 1, 'small_1'),
        (large_size, 3, 'large_1'),
        (small_size, 1, 'small_2'),
        (large_size, 3, 'large_2'),
    ]
    
    print(f"\nWorkload configuration:")
    print(f"Homogeneous: {homogeneous[0][0]}MB x 4 jobs")
    print(f"Heterogeneous: {small_size}MB (small) and {large_size}MB (large)")
    
    # Run experiments
    results = {}
    
    results['homogeneous_sequential'] = run_experiment(
        "Homogeneous Sequential (Batch Arrival)",
        homogeneous,
        num_trials=3,
        concurrent=False
    )
    
    results['heterogeneous_sequential'] = run_experiment(
        "Heterogeneous Sequential (Batch Arrival)",
        heterogeneous,
        num_trials=3,
        concurrent=False
    )
    
    results['homogeneous_concurrent'] = run_experiment(
        "Homogeneous Concurrent (Staggered Arrival)",
        homogeneous,
        num_trials=3,
        concurrent=True
    )
    
    results['heterogeneous_concurrent'] = run_experiment(
        "Heterogeneous Concurrent (Staggered Arrival)",
        heterogeneous,
        num_trials=3,
        concurrent=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for scenario_name, df in results.items():
        filename = f"results_{scenario_name}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Workload Measurements - Turnaround Time Analysis', fontsize=16)
    
    for idx, (scenario_name, df) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        
        if 'status' in df.columns:
            df_success = df[df['status'] == 'success']
        else:
            df_success = df
        
        if not df_success.empty and 'turnaround_time' in df_success.columns:
            df_success.boxplot(column='turnaround_time', by='workload_id', ax=ax)
            ax.set_title(scenario_name.replace('_', ' ').title())
            ax.set_xlabel('Workload')
            ax.set_ylabel('Turnaround Time (sec)')
            plt.sca(ax)
            plt.xticks(rotation=45)
        else:
            ax.text(0.5, 0.5, 'No successful runs',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(scenario_name.replace('_', ' ').title())
    
    plt.tight_layout()
    plot_filename = f'gpu_measurements_{timestamp}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nVisualization saved: {plot_filename}")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)