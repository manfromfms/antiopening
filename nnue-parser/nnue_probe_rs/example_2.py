import nnue_parser
nnue_parser.init_nnue_py("final.jnn")
wert = nnue_parser.eval_nnue_py("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1")
print("Bewertung in Centipawns:", wert)
input()
