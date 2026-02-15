"""
Parallel Compilation Support

Enables parallel compilation of independent modules and files
using multiprocessing and concurrent.futures.
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import threading
import queue
import time


@dataclass
class CompilationTask:
    """Represents a compilation task."""
    id: str
    input_file: str
    output_file: Optional[str] = None
    options: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CompilationResult:
    """Result of a compilation task."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    
    def __str__(self) -> str:
        if self.success:
            return f"Task {self.task_id}: SUCCESS ({self.duration:.3f}s)"
        else:
            return f"Task {self.task_id}: FAILED - {self.error}"


class WorkerPool:
    """
    Manages a pool of worker processes/threads for parallel compilation.
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = True,
        max_queue_size: int = 100,
    ):
        """
        Initialize worker pool.
        
        Args:
            num_workers: Number of workers (defaults to CPU count)
            use_processes: Use processes instead of threads
            max_queue_size: Maximum task queue size
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.max_queue_size = max_queue_size
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        self._active = True
        self._lock = threading.RLock()
    
    def submit(self, func: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to the worker pool.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future representing the task
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("Worker pool has been shut down")
            return self.executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Map a function over a list of items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            
        Returns:
            List of results
        """
        futures = [self.submit(func, item) for item in items]
        return [f.result() for f in futures]
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the worker pool.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        with self._lock:
            if self._active:
                self.executor.shutdown(wait=wait)
                self._active = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


class ParallelCompiler:
    """
    Manages parallel compilation of multiple files/modules.
    
    Features:
    - Dependency resolution
    - Task scheduling
    - Progress tracking
    - Error handling
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        use_processes: bool = False,  # Use threads by default for better compatibility
    ):
        self.pool = WorkerPool(num_workers=num_workers, use_processes=use_processes)
        self.tasks: Dict[str, CompilationTask] = {}
        self.results: Dict[str, CompilationResult] = {}
        self._lock = threading.RLock()
    
    def add_task(self, task: CompilationTask):
        """
        Add a compilation task.
        
        Args:
            task: CompilationTask to add
        """
        with self._lock:
            self.tasks[task.id] = task
    
    def compile_all(
        self,
        compile_func: Callable[[CompilationTask], Tuple[bool, Any, Optional[str]]],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, CompilationResult]:
        """
        Compile all tasks respecting dependencies.
        
        Args:
            compile_func: Function that compiles a task
                         Should return (success, output, error)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping task_id to CompilationResult
        """
        with self._lock:
            # Build dependency graph
            dep_graph = self._build_dependency_graph()
            
            # Topologically sort tasks
            sorted_tasks = self._topological_sort(dep_graph)
            
            # Group tasks by dependency level for parallel execution
            task_levels = self._group_by_level(sorted_tasks, dep_graph)
        
        # Compile each level in parallel
        total_tasks = len(self.tasks)
        completed = 0
        
        for level, task_ids in enumerate(task_levels):
            # Submit all tasks at this level
            futures = {}
            for task_id in task_ids:
                task = self.tasks[task_id]
                future = self.pool.submit(self._compile_task, task, compile_func)
                futures[future] = task_id
            
            # Wait for all tasks at this level to complete
            for future in as_completed(futures):
                task_id = futures[future]
                result = future.result()
                
                with self._lock:
                    self.results[task_id] = result
                
                completed += 1
                if progress_callback:
                    progress_callback(task_id, completed / total_tasks)
        
        return self.results
    
    def _compile_task(
        self,
        task: CompilationTask,
        compile_func: Callable,
    ) -> CompilationResult:
        """
        Compile a single task.
        
        Args:
            task: Task to compile
            compile_func: Function to perform compilation
            
        Returns:
            CompilationResult
        """
        start_time = time.perf_counter()
        
        try:
            success, output, error = compile_func(task)
            duration = time.perf_counter() - start_time
            
            return CompilationResult(
                task_id=task.id,
                success=success,
                output=output,
                error=error,
                duration=duration,
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            return CompilationResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration=duration,
            )
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build dependency graph from tasks.
        
        Returns:
            Dictionary mapping task_id to list of dependency task_ids
        """
        graph = {}
        for task_id, task in self.tasks.items():
            graph[task_id] = task.dependencies.copy()
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """
        Topologically sort tasks by dependencies.
        
        Args:
            graph: Dependency graph
            
        Returns:
            List of task IDs in dependency order
        """
        # Calculate in-degrees
        in_degree = {task_id: 0 for task_id in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Find tasks with no dependencies
        queue_tasks = [task_id for task_id, deg in in_degree.items() if deg == 0]
        sorted_tasks = []
        
        while queue_tasks:
            task_id = queue_tasks.pop(0)
            sorted_tasks.append(task_id)
            
            # Update in-degrees of dependents
            for other_id, deps in graph.items():
                if task_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue_tasks.append(other_id)
        
        # Check for cycles
        if len(sorted_tasks) != len(graph):
            raise ValueError("Circular dependency detected in compilation tasks")
        
        return sorted_tasks
    
    def _group_by_level(
        self,
        sorted_tasks: List[str],
        graph: Dict[str, List[str]],
    ) -> List[List[str]]:
        """
        Group tasks by dependency level for parallel execution.
        
        Args:
            sorted_tasks: Topologically sorted task IDs
            graph: Dependency graph
            
        Returns:
            List of task groups, where each group can be compiled in parallel
        """
        levels = []
        processed = set()
        
        for task_id in sorted_tasks:
            # Find the level for this task
            deps = graph[task_id]
            dep_levels = [self._find_level(dep, levels) for dep in deps if dep in processed]
            level = max(dep_levels, default=-1) + 1
            
            # Add to appropriate level
            while len(levels) <= level:
                levels.append([])
            levels[level].append(task_id)
            processed.add(task_id)
        
        return levels
    
    def _find_level(self, task_id: str, levels: List[List[str]]) -> int:
        """Find which level a task is in."""
        for i, level in enumerate(levels):
            if task_id in level:
                return i
        return -1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of compilation results.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total = len(self.results)
            successful = sum(1 for r in self.results.values() if r.success)
            failed = total - successful
            total_time = sum(r.duration for r in self.results.values())
            
            return {
                "total_tasks": total,
                "successful": successful,
                "failed": failed,
                "total_time": total_time,
                "average_time": total_time / total if total > 0 else 0,
                "parallelization_efficiency": self._calculate_efficiency(),
            }
    
    def _calculate_efficiency(self) -> float:
        """
        Calculate parallelization efficiency.
        
        Returns:
            Efficiency ratio (0-1), where 1 is perfect parallelization
        """
        if not self.results:
            return 0.0
        
        # Sum of all task durations
        sequential_time = sum(r.duration for r in self.results.values())
        
        # Actual wall time (max end time - min start time)
        # Approximation: max duration in any level
        actual_time = max(r.duration for r in self.results.values())
        
        if actual_time == 0:
            return 1.0
        
        return min(1.0, sequential_time / (actual_time * self.pool.num_workers))
    
    def shutdown(self):
        """Shutdown the parallel compiler."""
        self.pool.shutdown(wait=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def compile_modules_parallel(
    modules: List[str],
    compile_func: Callable[[str], Tuple[bool, Any, Optional[str]]],
    num_workers: Optional[int] = None,
) -> Dict[str, CompilationResult]:
    """
    Convenience function to compile multiple modules in parallel.
    
    Args:
        modules: List of module paths
        compile_func: Function that compiles a module
        num_workers: Number of worker threads/processes
        
    Returns:
        Dictionary mapping module to CompilationResult
    """
    compiler = ParallelCompiler(num_workers=num_workers, use_processes=False)
    
    # Create tasks
    for module in modules:
        task = CompilationTask(
            id=module,
            input_file=module,
            dependencies=[],
        )
        compiler.add_task(task)
    
    # Compile wrapper
    def task_compile_func(task: CompilationTask):
        return compile_func(task.input_file)
    
    # Compile all
    results = compiler.compile_all(task_compile_func)
    compiler.shutdown()
    
    return results


def estimate_optimal_workers() -> int:
    """
    Estimate optimal number of workers based on system resources.
    
    Returns:
        Recommended number of workers
    """
    cpu_count = mp.cpu_count()
    
    # For compilation workloads, typically use CPU count
    # But cap at reasonable maximum
    return min(cpu_count, 16)
