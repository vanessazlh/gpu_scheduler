"""
Workload definitions for GPU scheduler simulator.

This module defines the Job class representing GPU compute workloads
(e.g., deep learning training jobs) with memory requirements.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Job:
    """Represents a GPU compute workload (e.g., a deep learning model)."""
    
    job_id: int
    arrival_time: float  # When job arrives in the system
    execution_time: float  # Total GPU time needed (in milliseconds)
    memory_required: int  # Memory needed in MB
    priority: int = 1  # Job priority (default 1)
    
    # Runtime tracking
    remaining_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    virtual_runtime: float = 0.0  # For fair scheduler
    is_loaded: bool = False  # Whether model is currently in GPU memory
    num_swaps: int = 0  # Number of times swapped in/out
    
    def __post_init__(self):
        if self.remaining_time is None:
            self.remaining_time = self.execution_time
    
    def __repr__(self):
        return (f"Job(id={self.job_id}, arrival={self.arrival_time:.1f}ms, "
                f"exec={self.execution_time:.1f}ms, mem={self.memory_required}MB)")
    
    def is_complete(self) -> bool:
        """Check if job has finished execution."""
        return self.remaining_time <= 0
    
    def turnaround_time(self) -> Optional[float]:
        """Calculate turnaround time (completion - arrival)."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time
    
    def waiting_time(self) -> Optional[float]:
        """Calculate waiting time (turnaround - execution)."""
        if self.completion_time is None:
            return None
        return self.turnaround_time() - self.execution_time