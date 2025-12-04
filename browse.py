import chess
import sqlite3
import chess.svg
import chess.variant
import chess.polyglot
from flask import Flask, Response

from src.position import get_node_from_db


def generate_hash(board):
    hash = chess.polyglot.zobrist_hash(board)
    
    return hash if hash < (1 << 63) else hash - (1 << 64)


app = Flask(__name__)

@app.route("/position/<path:fen>")
def position_svg(fen):
    try:
        board = chess.variant.AntichessBoard(fen)
    except ValueError:
        return Response("Invalid FEN", status=400)
    
    hash = generate_hash(board)

    connection = sqlite3.Connection('./tree.db')
    data = get_node_from_db(hash, connection.cursor())

    table = ''

    for move in board.legal_moves:
        board.push(move)
        child_hash = generate_hash(board)
        child_fen = board.fen()
        board.pop()

        child_data = get_node_from_db(child_hash, connection.cursor())

        link = f'<a href="/position/{child_fen}">{move.uci()}</a>'

        table += f'''
            <tr>
                <td>{link}</td>
                <td>{child_data[1]}</td>
                <td>{int(child_data[2]/child_data[1] * 100) if child_data[1] > 0 else -1}</td>
                <td>{int(child_data[3]/child_data[1] * 100) if child_data[1] > 0 else -1}</td>
                <td>{int(child_data[4]/child_data[1] * 100) if child_data[1] > 0 else -1}</td>
            </tr>
        '''

    svg = chess.svg.board(
        board,
        size=350,
    )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Chess Position</title>
    </head>
    <body>
        <h1>{hash}</h1>
        {svg}
        <table>
            <tr>
                <td>Move</td>
                <td>Visits</td>
                <td>White Win</td>
                <td>Black Win</td>
                <td>Draw</td>
            </tr>
            <tr>
                <td>.</td>
                <td>{data[1]}</td>
                <td>{int(data[2]/data[1] * 100) if data[1] > 0 else -1}</td>
                <td>{int(data[3]/data[1] * 100) if data[1] > 0 else -1}</td>
                <td>{int(data[4]/data[1] * 100) if data[1] > 0 else -1}</td>
            </tr>
            {table}
        </table>
    </body>
    </html>
    """

    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
