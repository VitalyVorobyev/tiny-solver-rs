# Re-run every recorded tiny-solver pose-graph run on one side and compare sum_rho, grad_rel, solves.
# usage: python3 eval/recheck_pgo.py <old|new> <checkout with robust_pgo built> eval/results
import re, subprocess, sys, os
side, exe_dir, rdir = sys.argv[1], sys.argv[2], sys.argv[3]   # side: old|new
configs = {}
for f in ["pgo_grid_lm_default", "pgo_grid_gn_default", "pgo_outliers5", "pgo_huber_default", "pgo_huber_tight"]:
    for line in open(f"{rdir}/{f}.txt"):
        w = line.split()
        if not w or w[0] != side: continue
        if w[1].startswith("seed="): seed = w[1][5:]; w = w[2:]
        else: seed = "1"; w = w[1:]
        d, loss, scale, solver, tol, out = w[0], w[1], w[2], w[3], w[4], w[5][4:]
        rec = dict(kv.split("=", 1) for kv in w if "=" in kv)
        key = (d, loss, scale, solver, tol, out, seed)
        configs.setdefault(key, []).append((f, rec))
bad = 0
for key, recs in configs.items():
    d, loss, scale, solver, tol, out, seed = key
    env = dict(os.environ, RUST_LOG="tiny_solver=trace")
    p = subprocess.run(["./target/release/examples/robust_pgo", d, loss, scale, solver, tol, out, seed, "1"], cwd=exe_dir, env=env, capture_output=True, text=True)
    got = dict(kv.split("=", 1) for kv in p.stdout.split() if "=" in kv)
    solves = len(re.findall(r"solve Ax=b", p.stderr))
    for f, rec in recs:
        diffs = [k for k in ("sum_rho", "grad_rel") if rec.get(k) != got.get(k)]
        if "solves" in rec and f.startswith("pgo_grid") and solver == "lm" and int(rec["solves"]) != solves: diffs.append(f"solves {rec['solves']}->{solves}")
        status = "same" if not diffs else "DIFF " + " ".join(f"{k}:{rec.get(k)}->{got.get(k)}" if ":" not in k and "->" not in k else k for k in diffs)
        bad += bool(diffs)
        print(side, f, *key, status, flush=True)
print(side, "configs", len(configs), "records with differences", bad, flush=True)
