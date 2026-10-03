#!/bin/bash
# Pose-graph comparison. Needs two checkouts with `cargo build --release --example robust_pgo`
# done in each, and the ceres programs built (see README.md).
# usage: OLD=<master checkout> NEW=<PR checkout> CERES=<ceres build dir> \
#        run_pgo_grid.sh <default|tight> <lm|gn> <dataset:loss:outliers:seed>...
tol=$1; solver=$2; shift 2
tmp=$(mktemp -d)
for c in "$@"; do
  IFS=: read d l o s <<< "$c"
  reps=1; [ "$o" = "0" ] && reps=3
  for w in OLD NEW; do
    dir=${!w}
    sol=$(cd $dir && RUST_LOG=tiny_solver=trace ./target/release/examples/robust_pgo $d $l 1.0 $solver $tol $o $s 1 2>&1 >/dev/null | grep -c "solve Ax=b")
    line=$(cd $dir && ./target/release/examples/robust_pgo $d $l 1.0 $solver $tol $o $s $reps 2>/dev/null)
    echo "$( [ $w = OLD ] && echo old || echo new) seed=$s $line solves=$sol"
  done
  if [ "$solver" = "lm" ]; then
    (cd $OLD && DUMP_PROBLEM=$tmp/p.txt ./target/release/examples/robust_pgo $d $l 1.0 lm $tol $o $s 1)
    c_out=$($CERES/pgo $tmp/p.txt $l 1.0 $tol $tmp/pt.txt 2>/dev/null)
    pt=$(cd $OLD && EVAL_POINT=$tmp/pt.txt ./target/release/examples/robust_pgo $d $l 1.0 lm $tol $o $s 1)
    echo "ceres seed=$s $pt $(echo $c_out | awk '{print "iterations="$4" time="$6" "$2}')"
  fi
done
rm -rf $tmp
