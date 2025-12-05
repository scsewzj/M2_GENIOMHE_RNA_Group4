#!/usr/bin/env python3
"""
Training script for RNA statistical potential (C3'–C3' distances).

Usage example:
    python training.py data/pdbs/*.pdb --out-dir potentials --round-decimals 3
"""

import os
import sys
import argparse
from math import log
from typing import Dict, List, Tuple

import numpy as np
from Bio import PDB

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Canonical base-pair labels required by the assignment
BASE_PAIRS = ['AA', 'AU', 'AC', 'AG', 'UU', 'UC', 'UG', 'CC', 'CG', 'GG']

# Map residue names in PDB to canonical A/U/C/G labels
RESIDUE_NAME_MAP = {
    'A': 'A', 'ADE': 'A', 'DA': 'A',
    'C': 'C', 'CYT': 'C', 'DC': 'C',
    'G': 'G', 'GUA': 'G', 'DG': 'G',
    'U': 'U', 'URA': 'U', 'DU': 'U',
}

# Mapping from sorted pair (e.g. "CU") to canonical label (e.g. "UC")
SORTED_PAIR_TO_CANONICAL = {
    'AA': 'AA',
    'AC': 'AC',
    'AG': 'AG',
    'AU': 'AU',
    'CC': 'CC',
    'CG': 'CG',
    'CU': 'UC',  # C,U → UC
    'GG': 'GG',
    'GU': 'UG',  # G,U → UG
    'UU': 'UU',
}

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RNA distance-based statistical potential "
                    "from a dataset of RNA PDB structures."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDB/MMCIF files and/or directories containing PDB/MMCIF files."
    )
    parser.add_argument(
        "--out-dir", "-o",
        default="potentials",
        help="Output directory for the 10 base-pair potential files (default: potentials)."
    )
    parser.add_argument(
        "--round-decimals", "-r",
        type=int,
        default=3,
        help="Number of decimal places to round scores (default: 3)."
    )
    parser.add_argument(
        "--min-distance", "-mind",
        type=float,
        default=2.0,
        help="Minimum distance threshold in Angstroms. Distances below this are "
             "considered steric clashes and will be skipped with a warning (default: 2.0)."
    )
    parser.add_argument(
        "--atom-type", "-a",
        type=str,
        default="C3'",
        help="Atom type to use for distance calculations (default: C3'). "
             "Common options: C3', C4', C5', P (phosphate), C1'."
    )
    parser.add_argument(
        "--density-method", "-dm",
        type=str,
        choices=["histogram", "kde"],
        default="histogram",
        help="Density estimation method (default: histogram). "
             "Options: 'histogram' (discrete binning) or 'kde' (Kernel Density Estimation, smoother). "
             "Note: KDE is not yet implemented and will fall back to histogram."
    )
    parser.add_argument(
        "--kde-bandwidth", "-kbw",
        type=float,
        default=0.5,
        help="Bandwidth for Gaussian kernel in KDE (default: 0.5 Angstroms). "
             "Only used if --density-method=kde. Smaller values = more detail, larger = smoother."
    )
    parser.add_argument(
        "--max-distance", "-maxd",
        type=float,
        default=20.0,
        help="Maximum distance to consider in Angstroms (default: 20.0). "
             "Distances beyond this threshold will be ignored."
    )
    parser.add_argument(
        "--min-seq-sep", "-s",
        type=int,
        default=4,
        help="Minimum sequence separation between residues (default: 4). "
             "Only pairs with |j - i| >= min-seq-sep are considered. "
             "This filters out residues connected by backbone bonds."
    )
    parser.add_argument(
        "--bin-width", "-w",
        type=int,
        default=1.0,
        help="Width of distance bins in Angstroms (default: 1.0). "
             "Defines the resolution of the discretization step. "
             "A value of 1.0 is suitable for coarse-grained (C3') models. "
             "Lower values provide higher resolution but require larger datasets to avoid sparse counts."
    )
    parser.add_argument(
        "--max-score", "-ms",
        type=float,
        default=10.0,
        help="Maximum penalty score for never-observed distances (default: 10.0). "
             "Used as a cap for pseudo-energy scores and for steric clash penalties."
    )
    parser.add_argument(
        "--interchain-distance", "-id",
        action="store_true",
        help="Whether to include inter-chain distances in the statistics (default: False). "
    )
    return parser.parse_args()


def collect_pdb_paths(inputs: List[str]) -> List[str]:
    """
    Expand a list of paths (files or directories) into a flat list of PDB file paths.
    Support mmcif now: with suffix .cif or .ent
    """
    pdb_paths: List[str] = []

    for path in inputs:
        if os.path.isdir(path):
            # all files in directory that look like PDBs
            for name in os.listdir(path):
                full = os.path.join(path, name)
                if os.path.isfile(full) and (
                    name.lower().endswith(".pdb") or name.lower().endswith(".ent") or name.lower().endswith(".cif")
                ):
                    pdb_paths.append(full)
        elif os.path.isfile(path):
            pdb_paths.append(path)
        else:
            print(f"[WARN] Input path not found: {path}", file=sys.stderr)

    return sorted(set(pdb_paths))


def calculate_ED(atom1: PDB.Atom.Atom, atom2: PDB.Atom.Atom) -> float:
    """Euclidean distance between two Biopython Atom objects."""
    v = atom1.coord - atom2.coord
    return float(np.linalg.norm(v))


def extract_c3_atoms(chain: PDB.Chain.Chain, atom_type: str = "C3'") -> List[Tuple[int, PDB.Atom.Atom, str, float]]:
    """
    Extract specific atoms and their residue types (A/U/C/G) from a chain.

    Args:
        chain: BioPython Chain object
        atom_type: Name of the atom to extract (default: "C3'")

    Returns:
        List of tuples: (sequence_index, atom, base_code, bfactor)
        where bfactor is the temperature factor (reliability indicator).
    """
    atoms_list: List[Tuple[int, PDB.Atom.Atom, str, float]] = []
    residues = list(chain)  # chain is iterable over residues

    for i, residue in enumerate(residues):
        res_name = residue.resname.strip()
        base_code = RESIDUE_NAME_MAP.get(res_name, '')
        if not base_code:
            continue

        try:
            target_atom = residue[atom_type]
        except KeyError:
            # Atom not found in this residue; skip
            continue

        bfactor = target_atom.get_bfactor()
        atoms_list.append((i, target_atom, base_code, bfactor, chain.id))

    return atoms_list


def canonical_pair(res1: str, res2: str) -> str:
    """
    Map unordered pair (res1, res2) to one of the 10 canonical base-pair labels.
    Returns "" if the pair is not one of the 10 we care about.
    """
    s = ''.join(sorted((res1, res2)))  # e.g. "CU", "AU", "GG"
    return SORTED_PAIR_TO_CANONICAL.get(s, "")


def update_distance_counts(
    c3_atoms: List[Tuple[int, PDB.Atom.Atom, str, float]],
    observed_counts: Dict[str, np.ndarray],
    reference_counts: np.ndarray,
    distance_bins: np.ndarray,
    min_distance: float = 2.0,
    max_distance: float = 20.0,
    min_seq_sep: int = 4,
    return_raw_distances: bool = False,
    include_interchain: bool = False,
):
    """
    Given the C3' atoms for one chain, update the global observed and reference
    distance counts.

    - Only pairs with |j - i| >= min_seq_sep are considered.
    - Only distances >= min_distance and < max_distance are considered.
    - Distances < min_distance trigger a warning (steric clash).
    - Binning: Uses provided distance_bins array.
    - B-factors are extracted but not currently used in scoring.
    """
    n = len(c3_atoms)
    if not return_raw_distances:
        n_bins = len(distance_bins) - 1
    
    if return_raw_distances:
        raw_distances = {}
        for bp in BASE_PAIRS:
            raw_distances[bp] = []

    for idx1 in range(n):
        seq_idx1, atom1, res1, bfactor1, chain1 = c3_atoms[idx1]

        for idx2 in range(idx1 + 1, n):
            seq_idx2, atom2, res2, bfactor2, chain2 = c3_atoms[idx2]

            if chain1 == chain2:
                # Intra-chain distances
                # Sequence separation: |j - i| >= min_seq_sep
                if seq_idx2 - seq_idx1 < min_seq_sep:
                    continue
            else:
                # Inter-chain distances
                if include_interchain is False:
                    continue

            dist = calculate_ED(atom1, atom2)
            
            if return_raw_distances:
                bp = canonical_pair(res1, res2)
                if bp in raw_distances:
                    raw_distances[bp].append(dist)
            else:

                # Check for steric clash
                if dist < min_distance:
                    print(f"[WARN] Steric clash: {res1}{seq_idx1}-{res2}{seq_idx2} "
                      f"distance = {dist:.2f} Å (< {min_distance:.2f} Å threshold)",
                      file=sys.stderr)
                    continue

                # We restrict to [min_distance, max_distance) to stay inside valid range
                if dist >= max_distance:
                    continue

                # Find bin index
                bin_index = int(np.searchsorted(distance_bins, dist, side="right") - 1)
                if bin_index < 0 or bin_index >= n_bins:
                    continue

                bp = canonical_pair(res1, res2)
                if bp in observed_counts:
                    observed_counts[bp][bin_index] += 1

                # Reference counts: all pairs (X,X) regardless of type
                reference_counts[bin_index] += 1
    if return_raw_distances:
        return raw_distances


def compute_frequency(counts: np.ndarray, pseudo_count: float = 1e-12) -> np.ndarray:
    """
    Compute normalized frequencies from observation counts.

    Args:
        counts: Array of observation counts per bin
        pseudo_count: Small value to avoid numerical issues (default: 1e-12)

    Returns:
        Normalized frequency array, or None if no observations
    """
    total = float(counts.sum())
    if total <= 0.0:
        return None

    frequencies = counts.astype(float) / total
    # Avoid log(0) by clipping to pseudo_count minimum
    frequencies = np.clip(frequencies, pseudo_count, None)

    return frequencies


def compute_single_score(f_obs: float, f_ref: float, max_score: float) -> float:
    """
    Compute pseudo-energy score for a single distance bin.

    Formula: Score = -log(f_obs / f_ref)

    Args:
        f_obs: Observed frequency in this bin
        f_ref: Reference frequency in this bin
        max_score: Maximum penalty score (cap)

    Returns:
        Pseudo-energy score (capped at max_score)
    """
    if f_obs > 0.0 and f_ref > 0.0:
        val = -log(f_obs / f_ref)
        return min(val, max_score)
    else:
        return max_score


def compute_scores(
    observed_counts: Dict[str, np.ndarray],
    reference_counts: np.ndarray,
    n_bins: int,
    round_decimals: int,
    distance_bins: np.ndarray,
    min_distance: float,
    max_score: float = 10.0,
) -> Dict[str, List[float]]:
    """
    Compute pseudo-energy scores for each base pair and distance bin.

    Uses Boltzmann inversion: u_bar(i,j,r) = -log( f_obs(i,j,r) / f_ref(r) )
    where f_obs and f_ref are normalized frequencies.

    Bins corresponding to distances < min_distance are forced to max_score
    (steric clash penalty).

    Returns:
        Dictionary mapping base pairs to score lists: {bp: [score_bin_0, ..., score_bin_N-1]}
    """
    scores_dict: Dict[str, List[float]] = {}

    # Calculate bin centers to determine which bins are below min_distance
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2.0

    # Compute reference frequencies once (used for all base pairs)
    f_ref_bins = compute_frequency(reference_counts)

    if f_ref_bins is None:
        print("[WARN] Reference counts are all zero; something is wrong.",
              file=sys.stderr)
        # Fallback: uniform distribution
        f_ref_bins = np.ones(n_bins) / n_bins

    # Compute scores for each base pair
    for bp, counts in observed_counts.items():
        f_obs_bins = compute_frequency(counts)

        if f_obs_bins is None:
            # No observations at all for this pair
            scores_dict[bp] = [max_score] * n_bins
            continue

        # Compute score for each bin using helper function
        bp_scores = []
        for r in range(n_bins):
            # Force steric clash bins (distance < min_distance) to max_score
            if bin_centers[r] < min_distance:
                bp_scores.append(round(max_score, round_decimals))
            # Handle zero reference frequency case  
            elif reference_counts[r] == 0:
                bp_scores.append(round(max_score, round_decimals))
            else:
                score = compute_single_score(f_obs_bins[r], f_ref_bins[r], max_score)
                bp_scores.append(round(score, round_decimals))

        scores_dict[bp] = bp_scores

    return scores_dict


def save_scores(
    scores_dict: Dict[str, List[float]],
    distance_bins: np.ndarray,
    out_dir: str
) -> None:
    """
    Save one file per base pair with distance and score columns.

    Example output (with bin_width=1.0):
        # Distance(Å)  Score
        0.5  10.0
        1.5  10.0
        2.5  -0.3
        ...
        19.5  0.1

    Args:
        scores_dict: Dictionary mapping base pairs to score lists
        distance_bins: Array of bin edges (e.g., [0, 1, 2, ..., 20] for bin_width=1.0)
        out_dir: Output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    # Calculate bin centers (e.g., 0.5, 1.5, 2.5, ... for bin_width=1.0)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2.0

    for bp, scores in scores_dict.items():
        out_path = os.path.join(out_dir, f"{bp}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            # Write header
            f.write("# Distance(Å)  Score\n")
            # Write data
            for distance, score in zip(bin_centers, scores):
                f.write(f"{distance:.1f}  {score}\n")
        print(f"[INFO] Wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    pdb_files = collect_pdb_paths(args.inputs)
    if not pdb_files:
        print("[ERROR] No valid PDB/MMCIF files found.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(pdb_files)} PDB/MMCIF files.")

    # Check density method
    if args.density_method == "kde":
        print("[WARN] KDE is not yet fully implemented. Falling back to histogram method.",
              file=sys.stderr)
        print(f"[WARN] Ignoring --kde-bandwidth={args.kde_bandwidth}", file=sys.stderr)

    # Calculate distance bins dynamically based on max_distance
    num_bins = int((args.max_distance - 0.0) / args.bin_width) + 1
    distance_bins = np.linspace(0.0, args.max_distance, num_bins)
    n_bins = len(distance_bins) - 1

    print(f"[INFO] Distance range: 0.0 - {args.max_distance} Å")
    print(f"[INFO] Number of bins: {n_bins} (bin width: {args.bin_width} Å)")
    print(f"[INFO] Sequence separation: >= {args.min_seq_sep}")
    print(f"[INFO] Using atom type: {args.atom_type}")


    # Initialize global counts
    observed_counts = {
        bp: np.zeros(n_bins, dtype=np.int64) for bp in BASE_PAIRS
    }
    reference_counts = np.zeros(n_bins, dtype=np.int64)

    PDBparser = PDB.PDBParser(QUIET=True)
    CIFparser = PDB.MMCIFParser(QUIET=True)

    for pdb_path in pdb_files:
        print(f"[INFO] Processing {pdb_path}")
        if pdb_path.lower().endswith(".cif") or pdb_path.lower().endswith(".ent"):
            parser = CIFparser
        else:
            parser = PDBparser
        structure = parser.get_structure("RNA", pdb_path)

        for model in structure:
            model_atoms_all = []
            for chain in model:
                c3_atoms = extract_c3_atoms(chain, args.atom_type)
                model_atoms_all.extend(c3_atoms)

            if not model_atoms_all:
                continue

            update_distance_counts(
                model_atoms_all,
                observed_counts,
                reference_counts,
                distance_bins,
                args.min_distance,
                args.max_distance,
                args.min_seq_sep,
                include_interchain=args.interchain_distance
            )

    # Compute scores and save
    scores_dict = compute_scores(
        observed_counts,
        reference_counts,
        n_bins,
        args.round_decimals,
        distance_bins,
        args.min_distance,
        args.max_score
    )
    save_scores(scores_dict, distance_bins, args.out_dir)


if __name__ == "__main__":
    main()
