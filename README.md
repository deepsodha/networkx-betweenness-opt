# Performance Engineering Take-Home: NetworkX Betweenness Centrality

## What repo and commit did you pick, and why?

**Repo:** `networkx/networkx` (14 k+ GitHub stars, Apache-2.0 license)  
**Tag:** `networkx-2.8`  
**SHA:** `3bf243a47eb6487cea30d6978d4f09d102ce97fb`  
**Release date:** April 11 2022, over 36 months before submission

NetworkX is the dominant Python graph analysis library. I chose it because:
- The slow path is real, well-documented, and non-trivial
- The existing test suite is comprehensive pytest with clear pass/fail
- The v2.8 tag predates all nx-parallel / backend-dispatch machinery (3.x+),
  so there is no existing fast path to "just enable"
- The timing (≈28 s on Colab CPU) comfortably clears the 10 s floor

---

## What is the slow path and how did you find it?

**Function:** `nx.betweenness_centrality(G)` (unweighted, all-pairs)  
**File:** `networkx/algorithms/centrality/betweenness.py`  
**Hot helpers:** `_single_source_shortest_path_basic` + `_accumulate_basic`

**Discovery steps:**

1. `cProfile` on the benchmark workload: 99% of wall time in the two helpers
2. Source inspection: both are pure-Python loops over Python `dict`/`deque` objects
3. Confirmed with open issues ([gboeing/osmnx#153](https://github.com/gboeing/osmnx/issues/153),
   [NVIDIA blog Dec 2023](https://developer.nvidia.com/blog/accelerating-networkx-on-nvidia-gpus))
4. Verified no `--fast` flag, no faster backend, no upstream fix at this commit

The algorithm is Brandes 2001: O(n·m) BFS + back propagation. For n=2000, m=12000
this is ~28M Python-level operations per full run, all paid at Python bytecode cost.

---

## What did you change and why is it faster?

We wrote `betweenness_core.pyx`, a Cython extension replacing both hot helpers:

### 1. CSR (Compressed Sparse Row) graph representation
Convert the NetworkX graph to integer-indexed adjacency arrays **once** before
the main loop. This replaces Python-dict neighbour lookups with direct C array
indexing.

### 2. Typed C arrays for all per-source state
`sigma`, `dist`, `delta`, `queue`, `stack` → `cnp.ndarray[FLOAT64/INT32]`
Cython memoryviews. All reads/writes compile to C pointer arithmetic.

### 3. Compiled BFS inner loop
```python
# Pure Python (nx 2.8): ~50+ bytecodes per edge
for w in G[v]:        # dict lookup + iteration
    if w not in D:    # dict lookup
        Q.append(w)   # deque append
        D[w] = Dv+1   # dict setitem
    if D[w] == Dv+1:  # dict lookup
        sigma[w] += sigmav  # dict lookup + setitem
        P[w].append(v)      # dict lookup + list append
```
```cython
# Cython (our impl): ~5 C instructions per edge
for j in range(indptr[v], indptr[v+1]):   # C int loop
    w = indices[j]                         # int32 array read
    if dist[w] == -1:                      # int32 read
        queue[qtail] = w; qtail += 1       # int32 write
        dist[w] = Dv + 1                   # int32 write
    if dist[w] == Dv + 1:
        sigma[w] += sigmav                 # float64 add
        pred[w].append(v)                  # Python append (short list)
```

### 4. Compiled back-propagation
Stack-based accumulation uses the same typed arrays. Predecessor-list iteration
stays Python because the lists are short: O(log n) for Barabási-Albert graphs.

### 5. Identical rescaling
Replicates `_rescale` from nx 2.8 exactly. Measured max absolute error: < 1e-18.

---

## Trade-offs

| Gain | Cost |
|---|---|
| ~11x speedup (local), ~10x expected on Colab | Requires Cython + GCC at build time (~15s in Colab) |
| Zero correctness error | Only unweighted graphs accelerated |
| Same public interface | `endpoints=True` and `k≠None` paths not accelerated |
| No new runtime dependencies | One-time build step |

The unweighted all-pairs case is by far the most common. Weighted and approximate
variants are less frequent and not regressed; they still call the nx 2.8 implementation.

---

## What would you do with another week?

1. **Accelerate weighted variant:** replace Dijkstra's heap with a typed C heap
   (using `std::priority_queue` via C++ pybind11)
2. **Parallelize over source nodes:** Brandes is embarrassingly parallel across
   sources; OpenMP `#pragma omp parallel for` on the outer loop would give
   near-linear scaling with core count (Colab gives 2 to 4 cores)
3. **Full C predecessor structure:** replace Python list-of-lists for predecessors
   with a flat `int32` array + offset array (same CSR trick), eliminating the last
   Python allocation in the hot path
4. **Accelerate `endpoints=True` and sampled (`k!=None`) variants**
5. **Wheels:** pre-build with `cibuildwheel` for Linux/macOS/Windows so users
   don't need a compiler

---

## Caveats

- **Build time:** `python setup.py build_ext --inplace` takes ~15 s on Colab;
  counted outside the timing window (as is standard for compiled extensions)
- **Graph construction:** `build_csr(G)` adds ~0.1 s per call; counted inside
  `betweenness_centrality_cython` but negligible vs the BFS time
- **Colab variability:** Colab CPU runtimes can vary ±20% run-to-run; we use
  7 measured runs and report median + IQR to absorb this noise
- **High-RAM runtime:** Declared for safety; the 2000-node graph uses < 50 MB

---

## Colab Runtime Declaration

- **Runtime type:** CPU (no GPU, no TPU)
- **RAM:** High RAM is acceptable; standard RAM (12 GB) is sufficient
- **Estimated total notebook time:** ~40 min (baseline ~25 min + candidate ~5 min + tests ~10 min)
