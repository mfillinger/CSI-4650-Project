import time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def load_data(n_rows=100_000):
    path = Path("train.csv")
    if not path.exists():
        raise FileNotFoundError("train.csv is missing.")

    df = pd.read_csv(path)
    X = df.drop(columns=["Cover_Type", "Id"], errors="ignore").values.astype(np.float32)
    X_small = X[:n_rows]

    print(f"Loaded {X_small.shape[0]} samples with {X_small.shape[1]} features.")
    return X_small


def standardize_cpu(X):
    start = time.time()
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    Z = (X - mean) / std
    end = time.time()
    return Z, end - start


def standardize_gpu(X):
    if not HAS_GPU:
        raise RuntimeError("CuPy not installed or no GPU available.")

    Xg = cp.asarray(X)
    cp.cuda.Stream.null.synchronize()

    start = time.time()
    mean = Xg.mean(axis=0)
    std = Xg.std(axis=0)
    std = cp.where(std == 0, 1, std)
    Zg = (Xg - mean) / std
    cp.cuda.Stream.null.synchronize()
    end = time.time()

    return Zg, end - start


def main():
    X = load_data()

    #CPU benchmark
    Z_cpu, cpu_time = standardize_cpu(X)
    print(f"\nCPU time: {cpu_time:.4f} seconds")

    if HAS_GPU:
        Z_gpu, gpu_time = standardize_gpu(X)
        print(f"GPU time: {gpu_time:.4f} seconds")

        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")

        diff = np.abs(Z_cpu - cp.asnumpy(Z_gpu)).mean()
        print(f"Average difference CPU vs GPU: {diff:.8f}")
    else:
        print("\nNo GPU detected — only CPU benchmark ran.")


if __name__ == "__main__":
    main()
