# Parallel_Computing_Project

Small research / demo project showing single-process and parallel implementations
of simple OpenAI Gym-style environments (CartPole and Taxi).

Repository layout
- `cartPole.py` — single-process CartPole example
- `cartPole_parallel.py` — parallelized CartPole runner (multiprocessing)
- `taxi_run.py` — single-process Taxi example/runner
- `taxi_parallel.py` — parallelized Taxi runner (multiprocessing)

Requirements
- Python 3.8+
- pip
- Typical dependencies that these examples use: `gym` (or `gymnasium`), `numpy`.

Quick setup
1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

2. Install dependencies (example):

```powershell
pip install gym numpy
```

Running
- Run single-process CartPole:

```powershell
python cartPole.py
```

- Run parallel CartPole:

```powershell
python cartPole_parallel.py
```

- Run single-process Taxi:

```powershell
python taxi_run.py
```

- Run parallel Taxi:

```powershell
python taxi_parallel.py
```

Notes
- These scripts assume an environment compatible with OpenAI Gym APIs. If your code uses `gymnasium` or different versions, adjust the installs and imports accordingly.
- On Windows, multiprocessing spawn method is used by default; ensure the `if __name__ == "__main__":` guard is present in parallel scripts to avoid issues.

Contributing
- Open an issue or submit a pull request with improvements or fixes.

