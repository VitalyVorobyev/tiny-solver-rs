import collections, statistics, sys
E = sys.argv[1]
def load(path, solver=None):
    d = {}
    for line in open(path):
        f = line.split()
        if solver and f[3] != solver: continue
        key = (int(f[0]), int(f[1]), int(f[2]))
        kv = dict(x.split('=') for x in f if '=' in x)
        cost = None if f[4] == 'fail' else float(f[4])
        d[key] = (cost, float(kv.get('evals', 'nan')), float(kv.get('us', 'nan')))
    return d
runs = {}
for w in ('old', 'new'):
    for t in ('default', 'tight'):
        for s in ('lm', 'gn'):
            runs[(w, s, t)] = load(f'{E}/curves_{w}_{t}.txt', s)
for t in ('default', 'tight'):
    runs[('ceres', 'lm', t)] = load(f'{E}/curves_ceres_{t}.txt', 'ceres')
keys = sorted(runs[('old', 'lm', 'default')])
ref = {k: min(r[k][0] for r in runs.values() if r[k][0] is not None) for k in keys}
names = {0: 'Huber', 1: 'Cauchy', 2: 'Arctan'}
def summarize(run, subset):
    excess, fails, ev, us = [], 0, [], []
    for k in subset:
        c, e, u = run[k]
        ev.append(e); us.append(u)
        if c is None: fails += 1; continue
        excess.append(max(0.0, (c - ref[k]) / abs(ref[k])))
    n = len(subset)
    at = lambda tol: sum(x <= tol for x in excess)
    p90 = sorted(excess)[int(0.9 * len(excess))] if excess else float('nan')
    return f"{at(1e-9):4d} {at(1e-6):4d} {at(1e-3):4d} {n-len(excess)-0:3d}  med {statistics.median(excess):8.1e}  p90 {p90:8.1e}  max {max(excess):8.1e}  evals {statistics.mean(ev):6.1f}  ms {sum(us)/1000:8.1f}"
print("columns: within 1e-9 / 1e-6 / 1e-3 of the best known minimum, failures; relative excess of sum_rho; mean evaluations per fit; total ms (1 thread)")
for t in ('default', 'tight'):
    print(f"\n== {t} options")
    for loss in range(3):
        subset = [k for k in keys if k[2] == loss]
        print(f"-- {names[loss]} ({len(subset)} fits)")
        for v in (('old','lm',t), ('new','lm',t), ('old','gn',t), ('new','gn',t), ('ceres','lm',t)):
            if v[0] == 'ceres' or True:
                print(f"   {v[0]:5s} {v[1]}: {summarize(runs[v], subset)}")
    subset = [k for k in keys if k[1] == 0 and k[2] == 0]
    print(f"-- convex subset: line + Huber ({len(subset)} fits)")
    for v in (('old','lm',t), ('new','lm',t), ('ceres','lm',t)):
        print(f"   {v[0]:5s} {v[1]}: {summarize(runs[v], subset)}")
