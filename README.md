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

## Matroid-Constrained Diverse Matching (Problem 4)

The CLI now supports matroid constraints, allowing you to generate diverse matchings that satisfy additional structural requirements beyond simple matching constraints.

### Available Matroid Types

#### 1. **No Matroid** (Default - Problem 3)
Standard diverse matching without additional constraints.

```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3
```

**Graph Requirements:** Any weighted graph  
**Use Case:** Basic diverse matching scenario

#### 2. **Uniform Matroid** (`--matroid uniform`)
Constrains each matching to have at most `r` edges (automatically enforced by the `r` parameter).

```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 --matroid uniform
```

**Parameters:**
- Standard parameters only (`-k`, `-r`, `--delta`)

**Graph Requirements:** Any weighted graph  
**Use Case:** When you want explicit uniform matroid constraints in addition to matching constraints

**Mathematical Definition:** U(r, n) - independent sets are subsets of size ≤ r

#### 3. **Partition Matroid** (`--matroid partition`)
Partitions edges into groups based on weight threshold, with capacity limits per group.

```bash
python -m src.diverse_matching dataset -k 3 -r 15 --delta 0.3 \
  --matroid partition \
  --partition-threshold 15.0 \
  --partition-capacities 5 10
```

**Parameters:**
- `--partition-threshold <float>`: Weight value that divides edges into two groups
  - Group 0: edges with weight ≥ threshold (typically "high-value" collaborations)
  - Group 1: edges with weight < threshold (typically "low-value" collaborations)
- `--partition-capacities <cap0> <cap1>`: Maximum number of edges allowed from each group
  - First value: capacity for Group 0 (high-weight edges)
  - Second value: capacity for Group 1 (low-weight edges)

**Graph Requirements:**
- **Varied weights:** Graph must have both high and low-weight edges relative to the threshold
- **Sufficient capacity:** Sum of capacities must be ≥ r to ensure feasible matchings
- **Balanced distribution:** Ideally, bimodal weight distribution (e.g., ~50% high, ~50% low weights)

**Recommended Graph:** Generate with bimodal distribution:
```bash
python generate_graph.py --nodes 50 --edges 150 --type random \
  --weight-dist bimodal --output test_graph_partition
```

**Use Case:** Balance between high-value and low-value collaborations, prevent matchings from being dominated by one type of edge

**Mathematical Definition:** Partition matroid M = (E, I) where edges are partitioned into groups {G₀, G₁}, and a set S is independent iff |S ∩ Gᵢ| ≤ capacityᵢ for all i

**Example Output:**
```
🎯 Running Matroid-Constrained Diverse Matching (Problem 4)
   Matroid: Partition Matroid
   Threshold: 15.0
   Capacities: Group 0 (≥15.0) → 5, Group 1 (<15.0) → 10
   Partition: Group 0 has 76 edges, Group 1 has 724 edges
```

#### 4. **Graphic Matroid** (`--matroid graphic`)
Constrains each matching to form a forest (acyclic subgraph).

```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 --matroid graphic
```

**Parameters:**
- Standard parameters only (`-k`, `-r`, `--delta`)

**Graph Requirements:**
- **General graphs:** Works on any graph structure
- **Sparse graphs preferred:** Dense graphs may make finding acyclic matchings difficult
- **Tree-like structure:** Graphs with natural tree/forest structure work best

**Recommended Graph:** Generate with tree or grid structure:
```bash
python generate_graph.py --nodes 50 --edges 100 --type grid \
  --weight-dist normal --output test_graph_graphic
```

**Use Case:** When matchings must avoid cycles, useful for hierarchical collaboration structures

**Mathematical Definition:** Graphic matroid M = (E, I) where a set of edges S is independent iff S forms a forest (contains no cycles)

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
- Separate subgraph clusters for each matching
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
- `-k, --matchings`: Number of diverse matchings to generate
- `-r, --max-size`: Maximum edges per matching
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

**Example 1: Uniform Matroid with Visualization**
```bash
python -m src.diverse_matching smallw -k 3 -r 10 --delta 0.3 \
  --matroid uniform \
  --output uniform_matchings.dot \
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

**Example 3: Graphic Matroid (Forest Constraint)**
```bash
python -m src.diverse_matching dataset -k 3 -r 10 --delta 0.3 \
  --matroid graphic \
  --output forest_matchings.dot \
  --render-png
```

### Matroid Selection Guide

| Matroid | Constraint | Best Graph Type | Use Case |
|---------|-----------|-----------------|----------|
| **none** | Matching only | Any graph | Basic diverse matching |
| **uniform** | Size limit ≤ r | Any graph | Explicit size constraints |
| **partition** | Group capacities | Bimodal weights | Balance high/low-value edges |
| **graphic** | Acyclic (forest) | Sparse/tree-like | Hierarchical structures |

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
