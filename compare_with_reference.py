"""Compare a new run (ikc_outputs/tables) with the reference results in results/tables.
Usage (from the repository root, after running the notebook or script):  python compare_with_reference.py
Timing columns are machine-specific and are reported separately, not as differences."""
from pathlib import Path
import pandas as pd, numpy as np

REF, NEW = Path('results/tables'), Path('ikc_outputs/tables')
def is_timing(c):
    return 'time' in c or '_us' in c or c.endswith('_s')
if not NEW.exists():
    raise SystemExit('ikc_outputs/tables not found: run notebooks/IKC_certified_FK.ipynb or scripts/ikc_experiment.py first.')
lines, worst = [], 0.0
for ref_f in sorted(REF.glob('*.csv')):
    new_f = NEW / ref_f.name
    if not new_f.exists():
        lines.append(f'{ref_f.name}: MISSING in your run'); continue
    a, b = pd.read_csv(ref_f), pd.read_csv(new_f)
    if a.shape != b.shape:
        lines.append(f'{ref_f.name}: shape differs {a.shape} vs {b.shape}'); continue
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and not is_timing(c)]
    if not num:
        lines.append(f'{ref_f.name}: timing/text only (not compared)'); continue
    d = (a[num] - b[num]).abs()
    rel = d / a[num].abs().clip(lower=1e-12)
    m_abs, m_rel = float(np.nanmax(d.values)), float(np.nanmax(rel.values))
    worst = max(worst, m_rel)
    status = 'IDENTICAL' if m_abs == 0 else ('tiny difference' if m_rel < 0.01 else 'DIFFERS')
    lines.append(f'{ref_f.name}: {status} (max abs diff {m_abs:.4g}, max rel diff {100*m_rel:.2f}%)')
report = '\n'.join(lines) + f'\n\nLargest relative difference over all compared values: {100*worst:.2f}%\n'
print(report)
Path('ikc_outputs/comparison_with_reference.txt').write_text(report, encoding='utf-8')
print('Saved to ikc_outputs/comparison_with_reference.txt')
