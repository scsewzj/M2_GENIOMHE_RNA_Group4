from training import (
    extract_c3_atoms,
    calculate_ED,
    update_distance_counts,
    BASE_PAIRS,
    RESIDUE_NAME_MAP,
    SORTED_PAIR_TO_CANONICAL,
    load_chain_selection,
    resolve_allowed_chains,
)
from Bio import PDB

from plotting import load_all_potentials
import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score RNA structures using distance-based statistical potentials."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="PDB/mmCIF files and/or directories containing PDB/mmCIF files. NOTE: not supported with multiple directories or mixed files and directories."
    )
    parser.add_argument(
        "--atom-type", "-a",
        type=str,
        default="C3'",
        help="Atom type to use for distance calculations (default: C3'). "
             "Common options: C3', C4', C5', P (phosphate), C1'."
    )
    parser.add_argument(
        "--csv-output", "-o",
        type=str,
        default=None,
        help="If provided, output scores for each chain to the given CSV file. "
             "Columns: file,chain,score."
    )
    parser.add_argument(
        "--potentials-dir", "-p",
        type=str,
        default="data/potentials",
        help="Directory containing potential files (*.txt). Default: data/potentials"
    )
    parser.add_argument(
        "--chain-config", "-c",
        type=str,
        default=None,
        help="Optional config file listing which chains to process per structure. "
             "Each non-empty line should be '<filename> <chain_id>'. "
             "If omitted, all chains are processed."
    )
    return parser.parse_args()


def get_chain_distances(structure, atom_type="C3'", allowed_chains=None):
    """
    Collect raw distances per chain.
    Returns: dict[chain_id] -> dict[bp] -> List[float]
    """
    chain_distances = {}
    for model in structure:
        for chain in model:
            if allowed_chains is not None and chain.id not in allowed_chains:
                continue
            c3_atoms = extract_c3_atoms(chain, atom_type)
            if not c3_atoms:
                continue
            new_distances = update_distance_counts(
                c3_atoms, None, None, None, return_raw_distances=True
            )
            chain_distances[chain.id] = new_distances
    return chain_distances

def score_linear_interpolation(scoreprofile, distance):
    """
    Linear interpolation of score at given distance
    score: dict of pseudo energies for each basepair
    distance: float, interatomic distance
    """
    bins, scores = scoreprofile
    if distance <= bins[0]:
        return scores[0]
    if distance >= bins[-1]:
        return scores[-1]
    fraction = ((distance - bins[0]) / (bins[1] - bins[0])) % 1
    bin_index = int((distance - bins[0]) // (bins[1] - bins[0]))
    interpolated_score = scores[bin_index] + fraction * (scores[bin_index + 1] - scores[bin_index])
    return interpolated_score

def est_score(distances, potentials):
    total_score = 0.0
    for bp in BASE_PAIRS:
        if bp not in distances:
            continue
        dists = distances[bp]
        scoreprofile = potentials[bp]
        for dist in dists:
            score = score_linear_interpolation(scoreprofile, dist)
            total_score += score
    return total_score

def main():
    args = parse_args()
    print(args.input)
    potentials = load_all_potentials(args.potentials_dir)
    PDBparser = PDB.PDBParser(QUIET=True)
    CIFparser = PDB.MMCIFParser(QUIET=True)
    scores = {}
    joinflag = False
    chain_selection = load_chain_selection(args.chain_config)
    if chain_selection:
        print(f"[INFO] Loaded chain selection for {len(chain_selection)} entries.")

    if os.path.isdir(args.input[0]):
        inputs = [f for f in os.listdir(args.input[0]) if f.endswith('.pdb') or f.endswith('.cif')]
        joinflag = True
    else:
        inputs = args.input
    for input_file in inputs:
        fullpath = os.path.join(args.input[0], input_file) if joinflag else input_file
        print(f"[INFO] Processing {input_file}")
        if input_file.endswith('.pdb'):
            parser = PDBparser
        elif input_file.endswith('.cif'):
            parser = CIFparser
        else:
            print(f"[WARNING] Unsupported file format for {input_file}. Skipping.")
            continue
        structure = parser.get_structure("RNA", fullpath)
        allowed_chains = resolve_allowed_chains(fullpath, chain_selection)
        chain_distances = get_chain_distances(structure, args.atom_type, allowed_chains)
        if not chain_distances:
            print(f"[WARN] No chains processed for {input_file}.")
            continue
        scores[input_file] = {}
        for chain_id, dist_dict in chain_distances.items():
            score = est_score(dist_dict, potentials)
            print(f"Score for {input_file} (chain {chain_id}): {score}")
            scores[input_file][chain_id] = score
    if args.csv_output is not None:
        with open(args.csv_output, 'w') as f:
            print("file,chain,score", file=f)
            for input_file, chain_scores in scores.items():
                for chain_id, score in chain_scores.items():
                    print(f"{input_file},{chain_id},{score}", file=f)
        print(f"[INFO] Scores written to {args.csv_output}")
    
if __name__ == "__main__":
    main()
