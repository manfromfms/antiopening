import sys
import time
import tqdm
import chess
import click
import random
import sqlite3
import logging
import numpy as np
import chess.variant
import chess.polyglot


from src.position import setup_nodes_structure, get_node_from_db, write_node_into_db


# Setup logger
logging.basicConfig(level=logging.DEBUG, filename=f"{round(time.time() / 1000)}.log", filemode="w", format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Setup hash table
hash_table = np.zeros(shape=(1,1), dtype=np.int64)

# Setup constants
k1 = 1
k2 = 2


# Define hash generator
def generate_hash(board):
    hash = chess.polyglot.zobrist_hash(board)
    
    return hash if hash < (1 << 63) else hash - (1 << 64)


# Define main function
@click.command()
@click.option('--db', default='./tree.db', help='Database file path.')
@click.option('--hash', default=64, help='Size of hash table in megabytes.')
@click.option('--games', default=-1, help='Amount of games to simulate (<= 0 for no limit).')
def slave(db: str, hash: int, games: int):
    logger.info('Starting as slave')
    logger.info(f'Database file: {db}')
    connection = sqlite3.Connection(db)
    cursor = connection.cursor()

    hash_table_size = hash*1024*1024 // 40
    logger.info(f'Hash table size: {hash}mb ({hash_table_size} entries)')
    hash_table = np.zeros(shape=(5, hash_table_size), dtype=np.int64)

    setup_nodes_structure(cursor)
    connection.commit()

    if games > 0:
        for _ in tqdm.tqdm(range(1, games)):
            # Init board
            board = chess.variant.AntichessBoard()

            # Init array of visited positions
            visited = []
            hashes = set()

            # Each position will be checked in database if "do_lookup" is true
            do_lookup = True
            game_to_long = False
            while True:
                if len(visited) > 200:
                    game_to_long = True
                    break

                # Generate hash
                hash = generate_hash(board)

                rng = int(random.random() * 10000)

                # Check if position isn't in hash table
                if hash_table[0, hash % hash_table_size] != hash:
                    # Check db for current position if "do_lookup"
                    if do_lookup:
                        data = get_node_from_db(hash, cursor)
                        if data[0] == 0:
                            # Position wasn't found, so it's good to assume that none further neither will be
                            do_lookup = False
                        else:
                            # Position was found. Save previous position from table entry and overwrite it.
                            write_node_into_db(hash_table[:, hash % hash_table_size], cursor)
                            connection.commit()

                            hash_table[:, hash % hash_table_size] = data
                    
                    else:
                        # Without lookup just save current position from table and overwrite it.
                        write_node_into_db(hash_table[:, hash % hash_table_size], cursor)
                        connection.commit()

                        hash_table[:, hash % hash_table_size] = [hash, 0, 0, 0, 0]

                # Save position data to list of visited positions
                if hash not in hashes:
                    visited.append(hash_table[:, hash % hash_table_size])
                    hashes.add(hash)
                
                # Generate legal moves
                legal_moves = list(board.legal_moves)

                if len(legal_moves) == 0:
                    # The game has finished. Exit the loop.
                    break

                # If there are less visits then possible moves enter discovery mode
                if hash_table[1, hash % hash_table_size] < len(legal_moves):
                    picked = None

                    for i in range(len(legal_moves)):
                        index = (13669 * (i + rng) + 35317) % len(legal_moves)
                        board.push(legal_moves[index])
                        child_hash = generate_hash(board)

                        if hash_table[0, child_hash % hash_table_size] != child_hash:
                            data = get_node_from_db(child_hash, cursor)
                            if data[0] == 0:
                                picked = index
                                board.pop()
                                break

                        board.pop()

                    # FALLBACK if nothing new is found
                    if picked is None:
                        picked = random.randrange(len(legal_moves))

                    board.push(legal_moves[picked])


                else:
                    # If all children have been visited, find each one and apply weighted selection
                    min_weight = 100000
                    min_index = 0

                    for i in range(len(legal_moves)):
                        index = (13669 * (i + rng) + 35317) % len(legal_moves)
                        move = legal_moves[index]

                        # Do the move and get a child
                        board.push(move)
                        child_hash = generate_hash(board)
                        board.pop()
                        
                        data = np.zeros(5)

                        if hash_table[0, child_hash % hash_table_size] == child_hash:
                            # Child is in hash table
                            data = hash_table[:, child_hash % hash_table_size]
                        else:
                            # Try to find it in database
                            data = get_node_from_db(hash, cursor)

                        if data[0] == 0:
                            # Child wasn't found. Continue with it.
                            min_index = index
                            break

                        else:
                            # Child was found. Check if it should be visited instead.
                            parent_visits = max(1, hash_table[1, hash % hash_table_size])
                            child_visits = max(1, data[1])

                            exploitation = (float(data[2] - data[3]) / child_visits) * (1 if board.turn == chess.WHITE else -1)
                            exploration  = k2 * np.sqrt(np.log(parent_visits) / child_visits)

                            f = k1 * exploitation + exploration

                            if f < min_weight:
                                min_weight = f
                                min_index = index

                    # Continue with selected child
                    board.push(legal_moves[min_index])
                    continue
                            
            # After game finished check the result
            is_draw = board.is_variant_draw() or game_to_long
            is_white_win = not (board.is_variant_win() ^ (board.turn == chess.WHITE))

            for v in visited:
                v[1] += 1

                if is_draw:
                    v[4] += 1

                elif is_white_win:
                    v[2] += 1

                else:
                    v[3] += 1

                
                # Check if position isn't in hash table
                if hash_table[0, v[0] % hash_table_size] != v[0]:
                    data = get_node_from_db(v[0], cursor)
                    if data[0] == 0:
                        pass
                    else:
                        # Position was found. Save previous position from table entry and overwrite it.
                        write_node_into_db(hash_table[:, v[0] % hash_table_size], cursor)
                        connection.commit()

                        hash_table[:, v[0] % hash_table_size] = data
                else:
                    hash_table[:, v[0] % hash_table_size] = v
            

        # Dump hash table into db after all games are done
        for entry_index in range(hash_table_size):
            write_node_into_db(hash_table[:, entry_index], cursor)
        connection.commit()

# Execute main function
if __name__ == '__main__':
    slave()