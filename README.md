# CSI-4650-Project

This project was completed by Michael Fillinger & Fadi Jameel for CSI 4650 at Oakland University.

The goal of this project is to demonstrate how parallel computing can improve performance by comparing CPU and GPU execution times for a numerical standardization task.
The dataset used (train.csv) contains real-world environmental feature data, which is processed using NumPy on the CPU and CuPy on the GPU (if available).


## How to Run

1. Make sure the following files are in the same folder:
   `gpu_benchmark.py`
   `train.csv`
   `requirements_gpu.txt`

2. Install dependencies:
   ```bash
   pip install -r requirements_gpu.txt

3. Run the benchmark:
   ```bash
   python gpu_benchmark.py

4. View the results in the terminal
   The script will show
   ''bash
