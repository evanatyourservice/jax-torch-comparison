import jax
import torch
from jax_lm import benchmark_jax
from torch_lm import benchmark_torch

def run_comparison(trials=5, batch_size=128, seq_len=512, steps=30, warmup=1):
    print("=== System Information ===")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    print("\n")
    
    print("=== Running JAX Benchmark ===")
    jax_times = []
    for i in range(trials):
        ms = benchmark_jax(
            batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
            warmup=warmup
        )
        jax_times.append(ms)
        print(f"JAX Trial {i+1}: {ms:.2f} ms/step")
    
    print("\n=== Running PyTorch Benchmark ===")
    torch_times = []
    for i in range(trials):
        ms = benchmark_torch(
            batch_size=batch_size,
            seq_len=seq_len,
            steps=steps,
            warmup=warmup
        )
        torch_times.append(ms)
        print(f"PyTorch Trial {i+1}: {ms:.2f} ms/step")
    
    avg_jax = sum(jax_times) / len(jax_times)
    avg_torch = sum(torch_times) / len(torch_times)
    
    print("\n=== Results ===")
    print(f"JAX Average: {avg_jax:.2f} ms/step")
    print(f"PyTorch Average: {avg_torch:.2f} ms/step")
    print(f"Speed Ratio: {avg_torch/avg_jax:.2f}x (JAX is {avg_torch/avg_jax:.2f}x faster)")

if __name__ == "__main__":
    run_comparison() 