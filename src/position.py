import sqlite3
import numpy as np
from typing import Tuple

def setup_nodes_structure(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            hash INTEGER PRIMARY KEY,
            visits INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0
        );
    ''')


def get_node_from_db(hash: int, cursor: sqlite3.Cursor) -> np.ndarray:
    """Find node with given hash in database. 

    Args:
        hash (int): Hash value of the required node.
        cursor (sqlite3.Cursor): Cursor at the database.

    Returns:
        np.ndarray: Numpy array containing [hash, visits, wins, losses, draws].
    """

    if hash == 0:
        return np.zeros(5)

    cursor.execute('SELECT * FROM nodes WHERE hash = ?', (hash,))

    data = cursor.fetchone()
    if data is None:
        return np.zeros(shape=(5))

    return np.array(data)


def write_node_into_db(node: np.ndarray, cursor: sqlite3.Cursor) -> None:
    """
    Insert or update a node in the database.

    Args:
        node (np.ndarray): Array containing [hash, visits, wins, losses, draws].
        cursor (sqlite3.Cursor): Database cursor.
    """

    if node[0] == 0:
        return

    cursor.execute('''
        INSERT INTO nodes (hash, visits, wins, losses, draws)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(hash) DO UPDATE SET
            visits = excluded.visits,
            wins = excluded.wins,
            losses = excluded.losses,
            draws = excluded.draws;
    ''', (int(node[0]), int(node[1]), int(node[2]), int(node[3]), int(node[4])))