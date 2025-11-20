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

You can pass either a dataset folder (with an `edges.csv` file) or a single plain-text edge list. In both cases the loader expects each line to look like `u v weight`, ignoring blank lines and lines starting with `#`. Use `--edges-filename` if your folder stores edges under a different name. While running, the orchestrator prints `[Progress]` messages so you can see how many attempts were needed to lock in each matching. The final summary now lists each matching's size and total weighted distance, omitting the aggregate minimum distance.

## Diverse Matroid Bases (Problem 4)

The CLI now supports finding diverse bases of a matroid, as described in Problem 4 of the paper. This is distinct from Problem 3 (Diverse Matching). When a matroid is selected, the tool finds diverse **bases** (independent sets) of that matroid, which may or may not be matchings.

### Available Matroid Types

#### 1. **No Matroid** (Default - Problem 3)
Runs the standard **Diverse Matching** algorithm.
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3
```
**Output:** Diverse Matchings (sets of edges where no two share a node).

#### 2. **Uniform Matroid** (`--matroid uniform`)
Finds diverse sets of `r` edges (Problem 4).
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 --matroid uniform
```
**Output:** Diverse sets of size `r` (specifically, the top `r` weighted edges, subject to diversity penalties).
**Note:** These sets are NOT constrained to be matchings.

#### 3. **Partition Matroid** (`--matroid partition`)
Finds diverse bases respecting partition capacities.
```bash
python -m src.diverse_matching dataset -k 3 -r 15 --delta 0.3 \
  --matroid partition \
  --partition-threshold 15.0 \
  --partition-capacities 5 10
```
**Output:** Diverse sets of edges satisfying the group capacities.

#### 4. **Graphic Matroid** (`--matroid graphic`)
Finds diverse **Spanning Forests** (Problem 4).
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 --matroid graphic
```
**Output:** Diverse Spanning Forests (acyclic sets of edges).
**Note:** A spanning forest is rarely a matching. This correctly implements Problem 4 (Diverse Matroid Bases) rather than "Matchings that are Forests".

### Matroid Selection Guide

| Matroid | Problem | Output Structure | Use Case |
|---------|---------|------------------|----------|
| **none** | Problem 3 | **Matching** | Standard diverse collaboration matching |
| **uniform** | Problem 4 | **Set of size r** | Selecting top-k diverse edge sets |
| **partition** | Problem 4 | **Partitioned Set** | Balanced diverse edge selection |
| **graphic** | Problem 4 | **Spanning Forest** | Diverse network backbones (acyclic) |

### Visualization Output

Generate DOT files for Graphviz visualization:

```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 \
  --matroid partition --partition-threshold 15 --partition-capacities 3 10 \
  --output matchings.dot
```

**Auto-render to PNG** (requires Graphviz installed):
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 \
  --matroid uniform \
  --output matchings.dot \
  --render-png
```

**DOT File Features:**
- Separate subgraph clusters for each solution (matching or basis)
- Edge weights labeled on all edges
- For partition matroid: color-coded edges (red = Group 0, blue = Group 1)
- Legend explaining the visualization
- Matroid type displayed in graph title

**Manual rendering:**
```bash
dot -Tpng matchings.dot -o matchings.png
dot -Tsvg matchings.dot -o matchings.svg
dot -Tpdf matchings.dot -o matchings.pdf
```

### Complete Parameter Reference

**Required Arguments:**
- `dataset`: Path to dataset folder (with edges.csv) or edge list file
- `-k, --matchings`: Number of diverse solutions (matchings or bases) to generate
- `-r, --max-size`: Maximum edges per solution (for Problem 3) or size of basis (for Uniform Matroid)
- `--delta`: MWU accuracy parameter (0 < δ < 1)

**Optional Arguments:**
- `--seed`: Random seed for reproducibility
- `--max-iterations`: Hard cap on MWU iterations
- `--edges-filename`: Custom edges filename in dataset folder

**Matroid Arguments:**
- `--matroid {none,uniform,partition,graphic}`: Matroid type (default: none)
- `--partition-threshold`: Weight threshold for partition matroid
- `--partition-capacities`: Two integers for partition group capacities

**Visualization Arguments:**
- `--output`: Path to output DOT file
- `--render-png`: Auto-render DOT to PNG (requires Graphviz)

### Example Workflows

**Example 1: Uniform Matroid (Problem 4)**
```bash
python -m src.diverse_matching smallw -k 3 -r 10 --delta 0.3 \
  --matroid uniform \
  --output uniform_bases.dot \
  --render-png \
  --seed 42
```

**Example 2: Partition Matroid on Real Dataset**
```bash
python -m src.diverse_matching hep-th-1999 -k 3 -r 15 --delta 0.3 \
  --matroid partition \
  --partition-threshold 5.0 \
  --partition-capacities 5 10 \
  --output hep_partition.dot \
  --seed 42
```

**Example 3: Graphic Matroid (Spanning Forest)**
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 \
  --matroid graphic \
  --output forest_bases.dot \
  --render-png
```

### Troubleshooting

**Issue:** "Uniform weights don't work with MWU algorithm"  
**Solution:** Ensure your graph has varied edge weights. Use weight distributions like `normal`, `exponential`, or `bimodal` when generating test graphs.

**Issue:** "Partition matroid capacity too small"  
**Solution:** Ensure sum of capacities ≥ r. If your dataset has max matching size 100 but capacities sum to 15, increase capacities or reduce r.

**Issue:** "Large graphs take too long"  
**Solution:** For graphs with >10k edges, use smaller r values (e.g., r=10-15) or increase delta (e.g., 0.5) to reduce iterations.

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
