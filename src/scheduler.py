"""
GPU scheduler implementations.

This module provides different scheduling policies:
- FIFO: First-in-first-out
- FairScheduler: Virtual runtime-based fair scheduling (inspired by CFS)
- MemoryAwareScheduler: Fair scheduling that considers memory residency
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from src.workload import Job
from src.memory_manager import MemoryManager


class Scheduler(ABC):
    """Base class for GPU schedulers."""
    
    def __init__(self, gpu_memory: int = 8192):
        """
        Initialize scheduler.
        
        Args:
            gpu_memory: Total GPU memory in MB (default 8GB)
        """
        self.memory_manager = MemoryManager(gpu_memory)
        self.current_time = 0.0
        self.completed_jobs: List[Job] = []
        self.time_quantum = 100.0  # Time slice in milliseconds
    
    @abstractmethod
    def select_next_job(self, ready_queue: List[Job]) -> Optional[Job]:
        """
        Select the next job to execute.
        
        Args:
            ready_queue: List of jobs ready to execute
        
        Returns:
            Selected job or None if no job can be scheduled
        """
        pass
    
    def run_simulation(self, jobs: List[Job]) -> dict:
        """
        Run scheduling simulation with given jobs.
        
        Args:
            jobs: List of jobs to schedule
        
        Returns:
            Dictionary of simulation results and metrics
        """
        self.current_time = 0.0
        self.completed_jobs = []
        self.memory_manager.reset()
        
        # Create a copy of jobs to avoid modifying originals
        job_queue = [Job(
            job_id=j.job_id,
            arrival_time=j.arrival_time,
            execution_time=j.execution_time,
            memory_required=j.memory_required,
            priority=j.priority
        ) for j in jobs]
        
        ready_queue: List[Job] = []
        
        # Main simulation loop
        while job_queue or ready_queue:
            # Add newly arrived jobs to ready queue
            while job_queue and job_queue[0].arrival_time <= self.current_time:
                job = job_queue.pop(0)
                ready_queue.append(job)
            
            if not ready_queue:
                # No jobs ready, advance time to next arrival
                if job_queue:
                    self.current_time = job_queue[0].arrival_time
                continue
            
            # Select next job to run
            selected_job = self.select_next_job(ready_queue)
            
            if selected_job is None:
                # No job can be scheduled (shouldn't happen in most policies)
                self.current_time += 1.0
                continue
            
            # Load job into memory if needed
            if not self.memory_manager.is_loaded(selected_job):
                swap_time = self._load_job_with_eviction(selected_job, ready_queue)
                self.current_time += swap_time
            
            # Record start time if first execution
            if selected_job.start_time is None:
                selected_job.start_time = self.current_time
            
            # Execute job for one time quantum (or until completion)
            exec_time = min(self.time_quantum, selected_job.remaining_time)
            self.current_time += exec_time
            selected_job.remaining_time -= exec_time
            selected_job.virtual_runtime += exec_time
            
            # Check if job completed
            if selected_job.is_complete():
                selected_job.completion_time = self.current_time
                self.completed_jobs.append(selected_job)
                ready_queue.remove(selected_job)
                self.memory_manager.evict_job(selected_job)
            else:
                # Job preempted, stays in ready queue
                pass
        
        return self._compute_metrics()
    
    def _load_job_with_eviction(self, job: Job, ready_queue: List[Job]) -> float:
        """Load a job, evicting others if necessary."""
        total_time = 0.0
        
        # Make space if needed
        eviction_time = self.memory_manager.make_space_for(job, ready_queue)
        total_time += eviction_time
        
        # Load the job
        load_time = self.memory_manager.load_job(job)
        total_time += load_time
        
        return total_time
    
    def _compute_metrics(self) -> dict:
        """Compute performance metrics from completed jobs."""
        if not self.completed_jobs:
            return {}
        
        turnaround_times = [j.turnaround_time() for j in self.completed_jobs]
        waiting_times = [j.waiting_time() for j in self.completed_jobs]
        
        return {
            'total_jobs': len(self.completed_jobs),
            'total_time': self.current_time,
            'avg_turnaround_time': sum(turnaround_times) / len(turnaround_times),
            'avg_waiting_time': sum(waiting_times) / len(waiting_times),
            'total_swaps': self.memory_manager.total_swaps,
            'swap_frequency': self.memory_manager.total_swaps / len(self.completed_jobs)
        }


class FIFOScheduler(Scheduler):
    """First-In-First-Out scheduler."""
    
    def select_next_job(self, ready_queue: List[Job]) -> Optional[Job]:
        """Select job that arrived earliest."""
        if not ready_queue:
            return None
        
        # Sort by arrival time and return first
        ready_queue.sort(key=lambda j: j.arrival_time)
        return ready_queue[0]


class FairScheduler(Scheduler):
    """
    Fair scheduler based on virtual runtime (inspired by CFS).
    
    Selects the job with minimum virtual runtime to ensure fairness.
    """
    
    def select_next_job(self, ready_queue: List[Job]) -> Optional[Job]:
        """Select job with minimum virtual runtime."""
        if not ready_queue:
            return None
        
        # Select job with smallest virtual runtime
        return min(ready_queue, key=lambda j: j.virtual_runtime)


class MemoryAwareScheduler(Scheduler):
    """
    Memory-aware fair scheduler.
    
    Considers both virtual runtime (for fairness) and memory residency
    to minimize expensive swap operations.
    """
    
    def __init__(self, gpu_memory: int = 8192, memory_weight: float = 0.3):
        """
        Initialize memory-aware scheduler.
        
        Args:
            gpu_memory: Total GPU memory in MB
            memory_weight: Weight given to memory residency (0-1)
        """
        super().__init__(gpu_memory)
        self.memory_weight = memory_weight
    
    def select_next_job(self, ready_queue: List[Job]) -> Optional[Job]:
        """
        Select job considering both fairness and memory residency.
        
        Uses a weighted score: score = (1-w)*virtual_runtime - w*is_loaded
        Lower scores are scheduled first.
        """
        if not ready_queue:
            return None
        
        def compute_score(job: Job) -> float:
            # Normalize virtual runtime to [0, 1] range
            max_vruntime = max(j.virtual_runtime for j in ready_queue)
            norm_vruntime = job.virtual_runtime / max_vruntime if max_vruntime > 0 else 0
            
            # Memory bonus: loaded jobs get negative score component
            memory_bonus = -1.0 if self.memory_manager.is_loaded(job) else 0.0
            
            # Combined score
            return (1 - self.memory_weight) * norm_vruntime + self.memory_weight * memory_bonus
        
        # Select job with minimum score
        return min(ready_queue, key=compute_score)