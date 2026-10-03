# Evaluation: cost with loss functions (not part of the PR)

Supports the PR "Measure the cost with the loss functions". All reported costs are
sum_i rho(||r_i||^2), computed by the harness from the plain factor residuals with its own rho.
Ceres 2.2 solves the same objective and is used only as a reference.

## Setup

    # two checkouts: master (OLD) and the PR (NEW); copy examples/robust_*.rs into OLD
    cargo build --release --example robust_pgo --example robust_curves   # in both
    cmake -S eval/ceres -B /tmp/ceres-build -DCMAKE_PREFIX_PATH=$(brew --prefix)
    cmake --build /tmp/ceres-build

## Repro from the issue

    cargo run --release --example repro_b

## Pose graphs (section 1 and 3 of the PR)

    cfg=(); for d in input_M3500_g2o.g2o sphere2500.g2o parking-garage.g2o; do
      for l in huber cauchy; do
        cfg+=($d:$l:0:1 $d:$l:0.05:1 $d:$l:0.05:2 $d:$l:0.05:3 $d:$l:0.1:1); done; done
    OLD=... NEW=... CERES=/tmp/ceres-build eval/run_pgo_grid.sh default lm "${cfg[@]}" > lm.txt
    python3 eval/analyze_pgo.py lm.txt

`robust_pgo <file> <none|huber|cauchy> <scale> <gn|lm> <default|tight> <outlier_fraction> <seed>
<repeats>` adds `outlier_fraction * edges` wrong loop closures (identity measurement between random
poses more than 10 apart) and prints sum rho, the relative gradient and the best time.

## Curve fits (section 2)

    # in each checkout, single thread for timing
    RAYON_NUM_THREADS=1 ./target/release/examples/robust_curves 300 default > curves_old_default.txt
    RAYON_NUM_THREADS=1 ./target/release/examples/robust_curves 300 tight   > curves_old_tight.txt
    DUMP=1 ./target/release/examples/robust_curves 300 default 2> instances.txt > /dev/null
    /tmp/ceres-build/sweep default < instances.txt > curves_ceres_default.txt
    /tmp/ceres-build/sweep tight   < instances.txt > curves_ceres_tight.txt
    python3 eval/analyze_curves.py <dir with the six files>

## Failed linear solves (known limitation)

    # in each checkout: the same 300 fits without loss functions, tolerances off
    NOLOSS=1 ./target/release/examples/robust_curves 300 tight > curves_old_tight_noloss.txt
    grep -c ' lm fail' curves_old_tight_noloss.txt     # each fit is printed three times

Without loss functions the PR does not change what LM minimizes, and master already ends in a
failed linear solve in 27 of these 300 fits (23 with the PR; the cost is summed in a different
order, which changes the path on these fits). All of them are exponential fits that have no finite
minimum: b -> 0 while a and c grow without bound in opposite directions. LM keeps accepting steps,
u falls to about 1e-16 while J^T J becomes nearly singular, Cholesky fails, and LM returns `None`.

## Cost evaluation micro-benchmark (section 4)

    cargo run --release --example cost_bench                        # PR checkout only
    RAYON_NUM_THREADS=1 cargo run --release --example cost_bench

On one thread the two differ by about 15%. With the default thread pool the old evaluation is
slower than on one thread, because every residual block waits for the same mutex.

`results/` holds the raw outputs behind the PR's tables (measured with the deterministic layout of
the companion PR applied to both sides, so runs repeat exactly).
