import sys, re, collections
rows = collections.OrderedDict()
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) < 4 or f[0] not in ('old', 'new', 'ceres'): continue
    kv = dict(x.split('=', 1) for x in f if '=' in x)
    key = (f[2].replace('.g2o', '').replace('input_', '').replace('_g2o', ''), f[3], kv['out'], kv['seed'])
    rows.setdefault(key, {})[f[0]] = kv
print(f"{'dataset':15s} {'loss':6s} {'out':4s} {'seed':4s} | {'sum_rho old':>12s} {'new':>12s} {'ceres':>12s} | {'new vs old':>10s} | {'solves old/new/ceres':>20s} | {'time s old/new/ceres':>20s}")
better = worse = same = 0
for k, r in rows.items():
    o, n, c = (float(r[x]['sum_rho']) if x in r and 'sum_rho' in r[x] else float('nan') for x in ('old', 'new', 'ceres'))
    rel = (n - o) / o
    if abs(rel) < 1e-9: same += 1
    elif rel < 0: better += 1
    else: worse += 1
    sv = '/'.join(r[x].get('solves', r[x].get('iterations', '?')) for x in ('old', 'new', 'ceres') if x in r)
    tv = '/'.join(r[x].get('time', '?') for x in ('old', 'new', 'ceres') if x in r)
    print(f"{k[0]:15s} {k[1]:6s} {k[2]:4s} {k[3]:4s} | {o:12.6g} {n:12.6g} {c:12.6g} | {rel*100:+9.2f}% | {sv:>20s} | {tv:>20s}")
print(f"new vs old: lower {better}, higher {worse}, same {same}")
