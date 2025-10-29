"""
GPU scheduler simulator.

"""

import sys
sys.path.append('.')

from src.scheduler import FIFOScheduler, FairScheduler, MemoryAwareScheduler
from src.workload import Job


def main():
    print("="*60)
    print("GPU Scheduler Simulator - Quick Demo")
    print("="*60)
    
    # Create a simple workload
    jobs = [
        Job(job_id=0, arrival_time=0, execution_time=800, memory_required=4096),
        Job(job_id=1, arrival_time=100, execution_time=600, memory_required=3072),
        Job(job_id=2, arrival_time=200, execution_time=1000, memory_required=4096),
        Job(job_id=3, arrival_time=300, execution_time=500, memory_required=2048),
    ]
    
    print(f"\nWorkload: {len(jobs)} jobs")
    print("GPU Memory: 8192 MB (8GB)")
    for job in jobs:
        print(f"  {job}")
    
    # Run FIFO scheduler
    print("\n" + "-"*60)
    print("Running FIFO Scheduler...")
    print("-"*60)
    fifo = FIFOScheduler(gpu_memory=8192)
    fifo_results = fifo.run_simulation(jobs)
    
    print(f"Completed in: {fifo_results['total_time']:.2f}ms")
    print(f"Avg turnaround: {fifo_results['avg_turnaround_time']:.2f}ms")
    print(f"Total swaps: {fifo_results['total_swaps']}")
    
    # Run Fair scheduler
    print("\n" + "-"*60)
    print("Running Fair Scheduler...")
    print("-"*60)
    fair = FairScheduler(gpu_memory=8192)
    fair_results = fair.run_simulation(jobs)
    
    print(f"Completed in: {fair_results['total_time']:.2f}ms")
    print(f"Avg turnaround: {fair_results['avg_turnaround_time']:.2f}ms")
    print(f"Total swaps: {fair_results['total_swaps']}")
    
    # Run Memory-Aware scheduler
    print("\n" + "-"*60)
    print("Running Memory-Aware Scheduler...")
    print("-"*60)
    memory_aware = MemoryAwareScheduler(gpu_memory=8192, memory_weight=0.3)
    ma_results = memory_aware.run_simulation(jobs)
    
    print(f"Completed in: {ma_results['total_time']:.2f}ms")
    print(f"Avg turnaround: {ma_results['avg_turnaround_time']:.2f}ms")
    print(f"Total swaps: {ma_results['total_swaps']}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Scheduler':<20} {'Time (ms)':<15} {'Swaps':<10}")
    print("-"*60)
    print(f"{'FIFO':<20} {fifo_results['total_time']:<15.2f} {fifo_results['total_swaps']:<10}")
    print(f"{'Fair':<20} {fair_results['total_time']:<15.2f} {fair_results['total_swaps']:<10}")
    print(f"{'Memory-Aware':<20} {ma_results['total_time']:<15.2f} {ma_results['total_swaps']:<10}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == '__main__':
    main()