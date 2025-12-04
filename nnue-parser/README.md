# NNUE Parser Python Module

## Overview

The NNUE Parser Python Module is a high-performance library for evaluating chess positions using an Efficiently Updatable Neural Network (NNUE). Written in Rust and exposed to Python via PyO3 and maturin, this module is optimized for speed and efficiency—allowing you to use it entirely from Python.

## Features

- **High Performance:** Leverages SIMD optimizations for fast NNUE evaluations.
- **Python-Only Interface:** Use the NNUE parser directly from Python without any Rust code exposure.
- **Easy to Use:** Simple API for model initialization and evaluation using FEN strings.
- **Lightweight:** Designed for efficient memory and computational usage.

## Installation

Ensure you have Rust installed along with Python (3.7 or later). Then, install maturin:

```bash
pip install maturin
```

Clone the repository and build the module:

```bash
git clone https://github.com/github-jimjim/NNUE-Parser
cd nnue-parser
maturin develop
```

This command compiles the Rust code and installs the Python module in your current environment.

## Usage

Once installed, you can initialize the NNUE model and evaluate chess positions using FEN strings directly from Python:

```python
import nnue_parser

# Initialize the NNUE model (adjust the path to your NNUE file)
nnue_parser.init_nnue_py("path/to/nnue_file")

# Example FEN string for the starting chess position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Evaluate the position in centipawns
evaluation = nnue_parser.eval_nnue_py(fen)
print("Evaluation in centipawns:", evaluation)
```

## Benchmarking

Measure the Nodes per Second (NPS) by running the following benchmark test for one second:

```python
import time
import nnue_parser

def benchmark_nnue_nps():
    nnue_file_path = "path/to/nnue_file"

    print("Initializing NNUE model...")
    nnue_parser.init_nnue_py(nnue_file_path)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Warm-up phase
    print("Warming up the model...")
    for _ in range(10):
        _ = nnue_parser.eval_nnue_py(fen)

    print("Starting benchmark for 1 second...")
    count = 0
    start_time = time.perf_counter()
    end_time = start_time + 1.0  # Run for 1 second
    while time.perf_counter() < end_time:
        _ = nnue_parser.eval_nnue_py(fen)
        count += 1
    total_time = time.perf_counter() - start_time

    print(f"Evaluations in {total_time:.6f} seconds: {count}")
    nps = count / total_time
    print(f"NPS (Nodes per Second): {nps:.2f}")

if __name__ == "__main__":
    benchmark_nnue_nps()
```

## Project Structure

- **src/lib.rs:** Contains the NNUE parser implementation and PyO3 bindings.
- **Cargo.toml:** Configures Rust dependencies and the build process for PyO3 and maturin.
- **README.md:** This file.

## Contributing

Contributions are welcome! Please open issues or submit pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.

