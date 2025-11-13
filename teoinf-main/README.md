# Diverse Collaboration Matchings

This repository provides tooling to extract highly diverse collaboration matchings from the 1999 co-authorship datasets (astro-ph, cond-mat, hep-th). The goal is to select a set of \(k\) matchings (one-to-one collaboration snapshots) that maximizes the minimum weighted distance between any pair of matchings. The weighted distance between two matchings is the sum of the collaboration weights appearing in exactly one of them. The implementation follows the multiplicative-weights framework for Problem&nbsp;3 described in the attached paper excerpt.

## Installation

## Usage

The module exposes a functional API that mirrors Algorithms&nbsp;1–2 of the paper. Typical usage is to load a dataset into a `networkx.Graph`, build an edge-weight map, and invoke `orchestrate_diverse_matchings`.

```python
from pathlib import Path
import csv
import networkx as nx

from src.diverse_matching import (
	canonical_edge,
	orchestrate_diverse_matchings,
)

def load_dataset(folder: Path):
	graph = nx.Graph()
	weights = {}
	with (folder / "edges.csv").open() as fh:
		reader = csv.reader(fh)
		for u_raw, v_raw, w_raw in reader:
			u, v = int(u_raw), int(v_raw)
			w = float(w_raw)
			edge = canonical_edge(u, v)
			graph.add_edge(u, v, weight=w)
			weights[edge] = w
	return graph, weights

graph, weights = load_dataset(Path("astro-ph-1999"))
matchings, min_distance = orchestrate_diverse_matchings(
	graph,
	weights,
	k=5,
	r=20,
	delta=0.2,
	seed=42,
)
print(f"Minimum pairwise distance: {min_distance:.3f}")
```

Important parameters:

- `k`: number of matchings to output.
- `r`: maximum size (number of edges) allowed for every matching, as required by Problem&nbsp;3.
- `delta`: accuracy knob in the corollary; it sets the MWU learning rate and total number of oracle calls.
- `max_iterations` (optional): hard cap on the outer-loop iterations when you want an explicit budget.
- `seed` (optional): controls the randomness used when breaking ties.

```bash
python -m src.diverse_matching astro-ph-1999 -k 10 -r 20 --delta 0.2 --seed 7
```

You can pass either a dataset folder (with an `edges.csv` file) or a single plain-text edge list. In both cases the loader expects each line to look like `u v weight`, ignoring blank lines and lines starting with `#`. Use `--edges-filename` if your folder stores edges under a different name. While running, the orchestrator prints `[Progress]` messages so you can see how many attempts were needed to lock in each matching. The final summary now lists each matching’s size and total weighted distance, omitting the aggregate minimum distance.

## Testing

The script prints a summary of every selected matching, including size, total weight, and the top-weighted collaborations with scientist names when available.

## Testing

Run the unit tests with:

```bash
python -m pytest
```

## Approach Overview

1. Load the requested dataset and construct an undirected weighted graph of scientists.
2. Iteratively call the MWU-based matching routine: temporal weights down-weight edges that already appear in previous matchings, and the oracle (Edmonds' blossom) returns the current best candidate.
3. Collect \(k\) distinct matchings while maximising the minimum weighted symmetric-difference distance between every pair.
4. Report the resulting collection and the quality metrics.

## Requirements and installation (for VSCode)
From the project folder (/path/to/teoinf-main/PIA):

## Create and activate environment
Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Install dependencies (add numpy because the optimized code uses it):
pip install -r requirements.txt
pip install numpy

## How to run
Quick test with the small included dataset (small_test) to verify:
python -m src.diverse_matching small_test -k 2 -r 3 --delta 0.2 --seed 1 --approx greedy

Run in fast mode (recommended first) — use greedy (much faster, reads all the same):
python -m src.diverse_matching astro-ph-1999 -k 5 -r 20 --delta 0.2 --seed 42 --approx greedy

