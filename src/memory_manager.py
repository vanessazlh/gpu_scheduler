"""
GPU memory management for scheduler simulator.

Tracks which jobs are loaded in GPU memory and handles swap operations.
"""

from typing import Optional, Set
from src.workload import Job


class MemoryManager:
    """Manages GPU memory allocation and tracks loaded jobs."""
    
    # Swap time constants (in milliseconds)
    SWAP_IN_TIME = 300.0   # Time to load a model into GPU memory
    SWAP_OUT_TIME = 200.0  # Time to evict a model from GPU memory
    
    def __init__(self, total_memory: int):
        """
        Initialize memory manager.
        
        Args:
            total_memory: Total GPU memory capacity in MB
        """
        self.total_memory = total_memory
        self.used_memory = 0
        self.loaded_jobs: Set[int] = set()  # Set of job IDs currently in memory
        self.total_swaps = 0
    
    def available_memory(self) -> int:
        """Get currently available memory in MB."""
        return self.total_memory - self.used_memory
    
    def can_fit(self, job: Job) -> bool:
        """Check if job can fit in available memory."""
        if job.job_id in self.loaded_jobs:
            return True  # Already loaded
        return self.available_memory() >= job.memory_required
    
    def is_loaded(self, job: Job) -> bool:
        """Check if job is currently loaded in GPU memory."""
        return job.job_id in self.loaded_jobs
    
    def load_job(self, job: Job) -> float:
        """
        Load a job into GPU memory.
        
        Args:
            job: Job to load
        
        Returns:
            Time cost in milliseconds (0 if already loaded, SWAP_IN_TIME otherwise)
        
        Raises:
            MemoryError: If job cannot fit in available memory
        """
        if job.job_id in self.loaded_jobs:
            return 0.0  # Already loaded, no swap needed
        
        if not self.can_fit(job):
            raise MemoryError(
                f"Job {job.job_id} requires {job.memory_required}MB but only "
                f"{self.available_memory()}MB available"
            )
        
        self.loaded_jobs.add(job.job_id)
        self.used_memory += job.memory_required
        job.is_loaded = True
        job.num_swaps += 1
        self.total_swaps += 1
        
        return self.SWAP_IN_TIME
    
    def evict_job(self, job: Job) -> float:
        """
        Evict a job from GPU memory.
        
        Args:
            job: Job to evict
        
        Returns:
            Time cost in milliseconds (SWAP_OUT_TIME if was loaded, 0 otherwise)
        """
        if job.job_id not in self.loaded_jobs:
            return 0.0  # Not loaded, nothing to evict
        
        self.loaded_jobs.remove(job.job_id)
        self.used_memory -= job.memory_required
        job.is_loaded = False
        
        return self.SWAP_OUT_TIME
    
    def make_space_for(self, target_job: Job, jobs: list[Job]) -> float:
        """
        Evict jobs to make space for target job using LRU-like policy.
        
        Args:
            target_job: Job that needs to be loaded
            jobs: List of all jobs to consider for eviction
        
        Returns:
            Total time cost of evictions in milliseconds
        """
        if self.can_fit(target_job):
            return 0.0
        
        total_eviction_time = 0.0
        
        # Find jobs that are loaded but not running (candidates for eviction)
        eviction_candidates = [
            j for j in jobs 
            if j.job_id in self.loaded_jobs 
            and j.job_id != target_job.job_id
            and not j.is_complete()
        ]
        
        # Sort by virtual runtime (evict jobs that have run more)
        eviction_candidates.sort(key=lambda j: j.virtual_runtime, reverse=True)
        
        # Evict until we have enough space
        for job in eviction_candidates:
            if self.can_fit(target_job):
                break
            total_eviction_time += self.evict_job(job)
        
        return total_eviction_time
    
    def reset(self):
        """Reset memory manager to initial state."""
        self.used_memory = 0
        self.loaded_jobs.clear()
        self.total_swaps = 0