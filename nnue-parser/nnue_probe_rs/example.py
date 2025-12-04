import chess
import nnue_parser

nnue_parser.init_nnue_py("final.jnn")

def evaluate(board):
    return nnue_parser.eval_nnue_py(board.fen())

def minimax(board, depth, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluate(board), None

    best_move_found = None
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_value, _ = minimax(board, depth - 1, False)
            board.pop()
            if eval_value > max_eval:
                max_eval = eval_value
                best_move_found = move
        return max_eval, best_move_found
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_value, _ = minimax(board, depth - 1, True)
            board.pop()
            if eval_value < min_eval:
                min_eval = eval_value
                best_move_found = move
        return min_eval, best_move_found

def best_move(fen):
    board = chess.Board(fen)
    _, move = minimax(board, 3, True)
    if move is None:
        return "0000"  
    return move.uci()

def main():
    print("id name SimpleNNUE")
    print("id author Jimmy Luong")
    print("uciok")

    fen = chess.STARTING_FEN

    while True:
        command = input().strip()
        if command == "uci":
            print("id name SimpleNNUE")
            print("id author Jimmy Luong")
            print("uciok")
        elif command.startswith("position"):
            parts = command.split("position ")[-1]
            if parts.startswith("fen"):
                fen = parts.split("fen ")[-1]
            else:
                fen = chess.STARTING_FEN
        elif command.startswith("go"):
            move = best_move(fen)
            print(f"bestmove {move}")
        elif command == "quit":
            break

if __name__ == "__main__":
    main()
