import time
import nnue_parser

def benchmark_nnue_nps():
    nnue_file_path = "final.jnn"

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