"""
Memory monitoring utility for large-scale training

Tracks memory usage during initialization and training to help
determine if you need to request more RAM or reduce max_sent_len.
"""

import psutil
import os
import time

class MemoryMonitor:
    """Monitor and report memory usage"""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.checkpoints = []

    def get_memory_gb(self):
        """Get current memory usage in GB"""
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024**3)  # Convert bytes to GB

    def get_system_memory_gb(self):
        """Get total system memory and available memory"""
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        percent = mem.percent
        return total_gb, used_gb, available_gb, percent

    def checkpoint(self, label):
        """Record a memory checkpoint"""
        current_mem = self.get_memory_gb()
        total, used, available, percent = self.get_system_memory_gb()

        if current_mem > self.peak_memory:
            self.peak_memory = current_mem

        self.checkpoints.append({
            'label': label,
            'process_memory': current_mem,
            'system_used': used,
            'system_available': available,
            'system_percent': percent
        })

        print(f"\n{'='*70}")
        print(f"MEMORY CHECKPOINT: {label}")
        print(f"{'='*70}")
        print(f"Process memory:     {current_mem:6.2f} GB")
        print(f"System memory:      {used:6.2f} / {total:.2f} GB ({percent:.1f}% used)")
        print(f"Available:          {available:6.2f} GB")
        print(f"Peak so far:        {self.peak_memory:6.2f} GB")

        # Warning if getting close to limit
        if available < 50:
            print(f"⚠️  WARNING: Only {available:.1f} GB available!")
        if available < 20:
            print(f"🚨 CRITICAL: Risk of OutOfMemory crash!")

        print(f"{'='*70}\n")

    def summary(self):
        """Print summary of all checkpoints"""
        print(f"\n{'='*70}")
        print("MEMORY USAGE SUMMARY")
        print(f"{'='*70}")
        print(f"Peak memory usage: {self.peak_memory:.2f} GB")
        print(f"\nCheckpoint history:")
        print(f"{'Label':<40} {'Memory (GB)':<15} {'System %':<10}")
        print("-" * 70)
        for cp in self.checkpoints:
            print(f"{cp['label']:<40} {cp['process_memory']:>6.2f}         {cp['system_percent']:>6.1f}%")
        print(f"{'='*70}\n")


# Global monitor instance
_monitor = None

def start_monitoring():
    """Start memory monitoring"""
    global _monitor
    _monitor = MemoryMonitor()
    _monitor.checkpoint("Start")
    return _monitor

def checkpoint(label):
    """Record a checkpoint (convenience function)"""
    if _monitor is not None:
        _monitor.checkpoint(label)
    else:
        print("⚠️  Memory monitor not started! Call start_monitoring() first")

def summary():
    """Print summary (convenience function)"""
    if _monitor is not None:
        _monitor.summary()


# Example usage
if __name__ == "__main__":
    print(__doc__)
    print("\nExample usage in training script:")
    print("""
import monitor_memory

# Start monitoring
mem = monitor_memory.start_monitoring()

# Your code with checkpoints
mem.checkpoint("After imports")

hg = gsc.HarmonicGrammar(...)
mem.checkpoint("After HarmonicGrammar")

net = gsc.GscNet(...)
mem.checkpoint("After GscNet")

net.generate_corpus(...)
mem.checkpoint("After corpus generation")

net.train2(...)
mem.checkpoint("After first epoch")

# Summary at end
mem.summary()
""")
