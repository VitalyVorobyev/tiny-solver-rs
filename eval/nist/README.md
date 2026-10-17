# NIST StRD with tiny-solver's LM

[NIST StRD](https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml) has 27 nonlinear
regression problems with certified minima, each with two starts (54 runs).
`examples/nist.rs` is a port of Ceres 2.2's
[`examples/nist.cc`](https://github.com/ceres-solver/ceres-solver/blob/2.2.0/examples/nist.cc). It
has the same parser, the same models, and the same success rule: at least 4 correct digits (LRE) in
every parameter. `data/` holds Ceres' copies of the NIST files (`data/nist/` at 2.2.0).

```bash
RAYON_NUM_THREADS=1 cargo run --release --example nist -- eval/nist/data
```

To run it on another branch, copy `examples/nist.rs` into it. Each run prints:
- the problem, the start, and the options;
- the LRE and whether the run is solved;
- `none` if LM returned `None`;
- the accepted, rejected and failed-solve counts, which are read from LM's log messages;
- the final and certified cost;
- the time and the stopping reason.

There are two option sets:
- **tight:** the error thresholds are 0, and `max_iteration` is 10000;
- **default:** `OptimizerOptions::default()`.

## Results (`results/`)

| | solved, tight | mean LRE | attempts | rejected | failed solves | solved, default |
|---|---|---|---|---|---|---|
| master cfb0d77 | 53 / 54 | 7.89 | 2947 | 318 | 0 | 28 / 54 |
| `fix/lm-failed-solve` | 53 / 54 | 7.89 | 2885 | 205 | 0 | 28 / 54 |
| Ceres 2.2, `nist.cc`, dense QR | 53 / 54 | | | | | |

Both tiny-solver builds and Ceres fail the same run, BoxBOD from start 1.
