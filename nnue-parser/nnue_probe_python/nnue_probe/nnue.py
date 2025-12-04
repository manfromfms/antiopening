import numpy as np
import tensorflow as tf
import chess
from enum import IntFlag
from tensorflow import keras
from tensorflow.keras import layers

FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
SQUARE_NB = 64

_model = None

class PieceSquare(IntFlag):
    NONE     = 0
    W_PAWN   = 1
    B_PAWN   = 1 * SQUARE_NB + 1
    W_KNIGHT = 2 * SQUARE_NB + 1
    B_KNIGHT = 3 * SQUARE_NB + 1
    W_BISHOP = 4 * SQUARE_NB + 1
    B_BISHOP = 5 * SQUARE_NB + 1
    W_ROOK   = 6 * SQUARE_NB + 1
    B_ROOK   = 7 * SQUARE_NB + 1
    W_QUEEN  = 8 * SQUARE_NB + 1
    B_QUEEN  = 9 * SQUARE_NB + 1
    W_KING   = 10 * SQUARE_NB + 1
    END      = W_KING
    B_KING   = 11 * SQUARE_NB + 1
    END2     = 12 * SQUARE_NB + 1

    @staticmethod
    def from_piece(p: chess.Piece, is_white_pov: bool):
        return {
            chess.WHITE: {
                chess.PAWN: PieceSquare.W_PAWN,
                chess.KNIGHT: PieceSquare.W_KNIGHT,
                chess.BISHOP: PieceSquare.W_BISHOP,
                chess.ROOK: PieceSquare.W_ROOK,
                chess.QUEEN: PieceSquare.W_QUEEN,
                chess.KING: PieceSquare.W_KING
            },
            chess.BLACK: {
                chess.PAWN: PieceSquare.B_PAWN,
                chess.KNIGHT: PieceSquare.B_KNIGHT,
                chess.BISHOP: PieceSquare.B_BISHOP,
                chess.ROOK: PieceSquare.B_ROOK,
                chess.QUEEN: PieceSquare.B_QUEEN,
                chess.KING: PieceSquare.B_KING
            }
        }[p.color == is_white_pov][p.piece_type]

def orient(is_white_pov: bool, sq: int):
    return (63 * (not is_white_pov)) ^ sq

def make_halfkp_index(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
    return orient(is_white_pov, sq) + PieceSquare.from_piece(p, is_white_pov) + PieceSquare.END * king_sq

def get_halfkp_indeces(board: chess.Board):
    result = []
    for turn in [board.turn, not board.turn]:
        indices = []
        values = []
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices.append([0, make_halfkp_index(turn, orient(turn, board.king(turn)), sq, p)])
            values.append(1)
        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, 41024])
        result.append(tf.sparse.reorder(sparse_tensor))
    return result

def build_model_inputs():
    return (keras.Input(shape=(41024,), sparse=True, dtype=tf.int8),
            keras.Input(shape=(41024,), sparse=True, dtype=tf.int8))

def build_feature_transformer(inputs1, inputs2):
    ft_dense_layer = layers.Dense(FEATURE_TRANSFORMER_HALF_DIMENSIONS, name='feature_transformer')
    clipped_relu = layers.ReLU(max_value=127)
    transformed1 = clipped_relu(ft_dense_layer(inputs1))
    transformed2 = clipped_relu(ft_dense_layer(inputs2))
    return layers.Concatenate()([transformed1, transformed2])

def nnue_relu(x):
    return tf.maximum(0, tf.minimum(127, tf.dtypes.cast(tf.math.floordiv(x, 64), tf.int32)))

def build_hidden_layers(inputs):
    hidden_layer_1 = layers.Dense(32, name='hidden_layer_1')
    hidden_layer_2 = layers.Dense(32, name='hidden_layer_2')
    activation_1 = layers.Activation(nnue_relu)
    activation_2 = layers.Activation(nnue_relu)
    return activation_2(hidden_layer_2(activation_1(hidden_layer_1(inputs))))

def build_output_layer(inputs):
    output_layer = layers.Dense(1, name='output_layer')
    return output_layer(inputs)

def build_model():
    inputs1, inputs2 = build_model_inputs()
    outputs = build_output_layer(build_hidden_layers(build_feature_transformer(inputs1, inputs2)))
    return keras.Model(inputs=[inputs1, inputs2], outputs=outputs)

def nn_value_to_centipawn(nn_value):
    return (((nn_value // 16) * 100) // 208) / 100

def read_feature_transformer_parameters(nn_file, model):
    [header] = np.fromfile(nn_file, dtype=np.uint32, count=1)
    ASSOCIATED_HALFKP_KING = 1
    OUTPUT_DIMENSIONS = 2 * FEATURE_TRANSFORMER_HALF_DIMENSIONS
    HASH_VALUE = (0x5D69D5B9 ^ ASSOCIATED_HALFKP_KING) ^ OUTPUT_DIMENSIONS
    if header != HASH_VALUE:
        raise ValueError("Header passt nicht zum erwarteten Hash!")
    biases = np.fromfile(nn_file, dtype=np.int16, count=FEATURE_TRANSFORMER_HALF_DIMENSIONS)
    weights = np.fromfile(nn_file, dtype=np.int16, count=FEATURE_TRANSFORMER_HALF_DIMENSIONS * 41024)
    feature_transformer = model.get_layer(name='feature_transformer')
    feature_transformer.set_weights([weights.reshape((41024, -1)), biases])

def read_network_parameters(nn_file, model):
    _header = np.fromfile(nn_file, dtype=np.uint32, count=1)
    biases1 = np.fromfile(nn_file, dtype=np.int32, count=32)
    weights1 = np.fromfile(nn_file, dtype=np.int8, count=32 * 512)
    hidden_layer_1 = model.get_layer(name='hidden_layer_1')
    hidden_layer_1.set_weights([weights1.reshape((512, 32), order='F'), biases1])
    biases2 = np.fromfile(nn_file, dtype=np.int32, count=32)
    weights2 = np.fromfile(nn_file, dtype=np.int8, count=32 * 32)
    hidden_layer_2 = model.get_layer(name='hidden_layer_2')
    hidden_layer_2.set_weights([weights2.reshape((32, 32), order='F'), biases2])
    biases3 = np.fromfile(nn_file, dtype=np.int32, count=1)
    weights3 = np.fromfile(nn_file, dtype=np.int8, count=32)
    output_layer = model.get_layer(name='output_layer')
    output_layer.set_weights([weights3.reshape((32, 1), order='F'), biases3])
    current_pos = nn_file.tell()
    nn_file.seek(0, 2)
    if nn_file.tell() - current_pos != 0:
        raise ValueError("Es wurden nicht alle Parameter gelesen!")

def init_nnue(nnue_path: str):
    global _model
    with open(nnue_path, 'rb') as nn_file:
        version = np.fromfile(nn_file, dtype=np.uint32, count=1)[0]
        hash_value = np.fromfile(nn_file, dtype=np.uint32, count=1)[0]
        size = np.fromfile(nn_file, dtype=np.uint32, count=1)[0]
        arch = nn_file.read(size).decode()
        print(f'Version: {version}')
        print(f'Hash: {hash_value}')
        print(f'Architecture: {arch}')
        
        _model = build_model()
        read_feature_transformer_parameters(nn_file, _model)
        read_network_parameters(nn_file, _model)

def eval_nnue(fen: str) -> float:
    if _model is None:
        raise ValueError("Modell nicht initialisiert! Bitte zuerst init_nnue(nnue_path) aufrufen.")
    
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise ValueError(f"Ungültiger FEN-String: {e}")
    
    features = get_halfkp_indeces(board)
    prediction = _model.predict(features)[0][0]
    centipawn = nn_value_to_centipawn(prediction)
    return centipawn
