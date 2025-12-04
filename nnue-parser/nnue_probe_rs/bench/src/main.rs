mod nnue;
extern crate lazy_static;
extern crate once_cell;
extern crate byteorder;
extern crate chess;

use std::time::{Duration, Instant};
use nnue::{init_nnue, eval_nnue};

fn main() {
    let nnue_path = "final.jnn";
    if let Err(e) = init_nnue(nnue_path) {
        eprintln!("Fehler beim Initialisieren des Modells: {}", e);
        return;
    }

    let fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let fen2 = "2rk4/8/8/8/8/8/8/K7 w - - 0 1";

    match eval_nnue(fen1) {
        Ok(bewertung) => println!("Bewertung der Position 1: {}", bewertung),
        Err(e) => {
            eprintln!("Fehler beim Evaluieren der Position 1: {}", e);
            return;
        }
    }

    match eval_nnue(fen2) {
        Ok(bewertung) => println!("Bewertung der Position 2: {}", bewertung),
        Err(e) => {
            eprintln!("Fehler beim Evaluieren der Position 2: {}", e);
            return;
        }
    }

    let duration = Duration::from_secs(1);

    let start1 = Instant::now();
    let mut count1 = 0u64;
    while start1.elapsed() < duration {
        if eval_nnue(fen1).is_ok() {
            count1 += 1;
        } else {
            eprintln!("Fehler während der Evaluation von Position 1.");
            return;
        }
    }
    println!("Nodes per Second (nps) für Position 1: {}", count1);

    let start2 = Instant::now();
    let mut count2 = 0u64;
    while start2.elapsed() < duration {
        if eval_nnue(fen2).is_ok() {
            count2 += 1;
        } else {
            eprintln!("Fehler während der Evaluation von Position 2.");
            return;
        }
    }
    println!("Nodes per Second (nps) für Position 2: {}", count2);
}
