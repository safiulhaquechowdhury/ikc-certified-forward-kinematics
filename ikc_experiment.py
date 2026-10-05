"""Command-line version of notebooks/IKC_certified_FK.ipynb (identical code, cell markers kept as comments).
Run from the repository root:  python scripts/ikc_experiment.py
Outputs are written to ./ikc_outputs (figures, tables, run_log.txt, summary.json)."""
# %% [markdown]
# # Certified learned forward kinematics for parallel mechanisms
# **Inverse-kinematics consistency residuals and conformal pose-error bounds near Type II singularities**
#
# This notebook reproduces every number, table and figure of the manuscript. It is CPU-only and light
# (NumPy, SciPy, pandas, scikit-learn, Matplotlib); a full run takes a few minutes on a laptop.
#
# **Pipeline (both mechanisms: planar 3-RPR and 6-6 Gough-Stewart platform)**
# 1. Closed-form inverse kinematics (IK) and analytic IK Jacobian; finite-difference verification.
# 2. Synthetic data: poses sampled in one working mode (fixed sign of det J) with a conditioning floor,
#    actuator lengths corrupted by Gaussian sensor noise.
# 3. Learned forward kinematics (FK): multilayer perceptron (MLP) regressor, plus a 5-member deep ensemble.
# 4. IK-consistency residual, Newton-Raphson refinement and the first-order error estimator (IKC).
# 5. Split conformal prediction with six scale choices; conditional coverage by conditioning decile.
# 6. Distribution shift, consistency-versus-correctness (branch) analysis, ablations, timing.
#
# All outputs (figures, tables, logs, JSON summary) are written to `ikc_outputs/`.
# Save the executed notebook after the run so that the complete record (code + outputs) is kept.

# %%
import os, sys, time, json, math, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')   # headless when run as a script
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import scipy
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

CFG = {
    'seed': 2026,
    'sigma_q': 1.0e-5,            # actuator length noise std [m] (10 micrometre)
    'lci_min': 0.01,              # conditioning floor used to define the working region
    'n_train': 30000,             # FK training set
    'n_fit': 5000,                # split used only by the learned-scale baseline
    'n_pool': 25000,              # calibration + test pool (re-split in every repetition)
    'n_cal': 5000,                # calibration size per repetition
    'n_shift': 10000,             # extrapolation-band test set
    'n_reps': 20,                 # random calibration/test repetitions
    'alpha': 0.10,                # miscoverage level (90 % bounds)
    'n_ens': 5,                   # deep-ensemble members (member 0 is the point predictor)
    'hidden': (128, 128),         # MLP hidden layers
    'max_epochs': 200,
    'n_bins_mondrian': 5,         # strata for group-conditional (Mondrian) conformal
    'ncal_grid': (100, 250, 500, 1000, 2000, 5000),
    'ncal_reps': 50,
    'home_max_iter': 50,          # Newton-Raphson from the home pose (classical baseline)
    'newton_tol': 1e-10,          # residual tolerance [m]
    'converge_steps': 10,         # Newton steps for the converged (branch) analysis
    'timing_queries': 2000,
}
OUT = Path('ikc_outputs')
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
LOG = open(OUT / 'run_log.txt', 'w', encoding='utf-8')

def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.write(s + '\n'); LOG.flush()

T_START = time.time()
log('Run started', time.strftime('%Y-%m-%d %H:%M:%S'))
log('Python', sys.version.split()[0], '| numpy', np.__version__, '| scipy', scipy.__version__,
    '| pandas', pd.__version__, '| scikit-learn', sklearn.__version__, '| matplotlib', matplotlib.__version__)
log('Platform', platform.platform())
log('CFG', json.dumps(CFG))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5, 'legend.fontsize': 8.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.9, 'lines.linewidth': 1.6,
    'savefig.dpi': 300, 'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.28, 'grid.linewidth': 0.6,
})
METHODS = ['const', 'ens', 'learn', 'strat', 'ikc', 'ikc_m']
MNAME = {'const': 'CP-const', 'ens': 'CP-ens', 'learn': 'CP-learn', 'strat': 'CP-strat',
         'ikc': 'CP-IKC (proposed)', 'ikc_m': 'CP-IKC-M (proposed)'}
MCOL = {'const': '#7f7f7f', 'ens': '#1f77b4', 'learn': '#2ca02c', 'strat': '#9467bd',
        'ikc': '#d62728', 'ikc_m': '#ff7f0e'}
MMARK = {'const': 'o', 'ens': 's', 'learn': '^', 'strat': 'v', 'ikc': 'D', 'ikc_m': 'P'}

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUT / 'figures' / f'{name}.pdf', bbox_inches='tight')
    plt.show(); plt.close(fig)

def savetab(df, name):
    df.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
    log(f'\n[{name}]'); log(df.to_string(index=False))

# %% [markdown]
# ## 1. Rotation utilities and mechanism models
# Both mechanisms are leg-length actuated. The IK Jacobian maps the platform twist to actuator rates,
# $\dot{\mathbf q} = \mathbf J(\mathbf x)\,\boldsymbol\xi$, with row $i$ equal to
# $[\mathbf u_i^{\mathsf T},\ (\mathbf c_i\times\mathbf u_i)^{\mathsf T}]$ ($\mathbf u_i$ unit leg vector,
# $\mathbf c_i$ platform anchor in the base frame). A Type II singularity occurs where $\det\mathbf J=0$.

# %%
def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(np.shape(r) + (3, 3))
    R[..., 0, 0] = cy * cp; R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp; R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp;     R[..., 2, 1] = cp * sr;                R[..., 2, 2] = cp * cr
    return R

def R2rpy(R):
    p = -np.arcsin(np.clip(R[..., 2, 0], -1.0, 1.0))
    r = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    y = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([r, p, y], -1)

def expso3(w):
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = w / np.maximum(th, 1e-15)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]; K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]; K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)

class Planar3RPR:
    """Planar 3-RPR parallel manipulator; pose x = (x, y, phi), twist = (dx, dy, dphi)."""
    name, key, n = 'planar 3-RPR', 'rpr', 3
    pos, rot = slice(0, 2), slice(2, 3)
    def __init__(self):
        self.A = np.array([[0.00, 0.00], [1.00, 0.00], [0.42, 0.85]])      # base joints [m]
        self.B = np.array([[-0.17, -0.10], [0.17, -0.10], [0.00, 0.17]])   # platform joints [m]
        self.Lc = float(np.max(np.linalg.norm(self.B, axis=1)))           # characteristic length
        self.aspect = +1                                                   # working mode: det J > 0
        self.home = np.array([0.50, 0.375, 0.0])
        self.box = {'x': (0.20, 0.80), 'y': (0.15, 0.60), 'phi_deg': 50.0}
        self.shift_band_deg = (50.0, 60.0)
    def legs(self, X):
        c, s = np.cos(X[:, 2]), np.sin(X[:, 2])
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        cvec = np.einsum('nij,kj->nki', R, self.B)
        return X[:, None, :2] + cvec - self.A[None], cvec
    def ik(self, X):
        L, _ = self.legs(X); return np.linalg.norm(L, axis=-1)
    def jac(self, X):
        L, c = self.legs(X); u = L / np.linalg.norm(L, axis=-1, keepdims=True)
        cross = c[..., 0] * u[..., 1] - c[..., 1] * u[..., 0]
        return np.concatenate([u, cross[..., None]], -1)
    def retract(self, X, d):
        return X + d
    def pose_error(self, Xh, X):
        ep = np.linalg.norm(Xh[:, :2] - X[:, :2], axis=1)
        er = np.abs((Xh[:, 2] - X[:, 2] + np.pi) % (2 * np.pi) - np.pi)
        return ep, er
    def sample(self, rng, M, region='id'):
        x = rng.uniform(*self.box['x'], M); y = rng.uniform(*self.box['y'], M)
        if region == 'id':
            phi = np.deg2rad(rng.uniform(-self.box['phi_deg'], self.box['phi_deg'], M))
        else:
            lo, hi = self.shift_band_deg
            phi = np.deg2rad(rng.uniform(lo, hi, M) * rng.choice([-1.0, 1.0], M))
        return np.column_stack([x, y, phi])
    def in_box(self, X):
        b = self.box
        return ((X[:, 0] >= b['x'][0]) & (X[:, 0] <= b['x'][1]) & (X[:, 1] >= b['y'][0]) &
                (X[:, 1] <= b['y'][1]) & (np.abs(X[:, 2]) <= np.deg2rad(b['phi_deg'])))

class GoughStewart66:
    """6-6 Gough-Stewart platform; pose x = (x, y, z, roll, pitch, yaw), twist = (dp, omega)."""
    name, key, n = '6-6 Gough-Stewart', 'gsp', 6
    pos, rot = slice(0, 3), slice(3, 6)
    def __init__(self, Rb=0.50, Rp=0.30, ab_deg=12.0, ap_deg=12.0):
        ab, ap = np.deg2rad(ab_deg), np.deg2rad(ap_deg)
        tb = np.array([-ab, ab, 2*np.pi/3 - ab, 2*np.pi/3 + ab, 4*np.pi/3 - ab, 4*np.pi/3 + ab])
        tp = np.array([-np.pi/3 + ap, np.pi/3 - ap, np.pi/3 + ap, np.pi - ap, np.pi + ap, 5*np.pi/3 - ap])
        self.A = np.stack([Rb * np.cos(tb), Rb * np.sin(tb), np.zeros(6)], 1)
        self.B = np.stack([Rp * np.cos(tp), Rp * np.sin(tp), np.zeros(6)], 1)
        self.Rb, self.Rp, self.Lc = Rb, Rp, Rp
        self.aspect = -1                                                   # working mode: det J < 0
        self.home = np.array([0.0, 0.0, 0.55, 0.0, 0.0, 0.0])
        self.box = {'xy': 0.12, 'z': (0.45, 0.65), 'tilt_deg': 25.0, 'yaw_deg': 75.0}
        self.shift_band_deg = (25.0, 30.0)
    def R(self, X):
        return rpy2R(X[:, 3], X[:, 4], X[:, 5])
    def legs(self, X):
        c = np.einsum('nij,kj->nki', self.R(X), self.B)
        return X[:, None, :3] + c - self.A[None], c
    def ik(self, X):
        L, _ = self.legs(X); return np.linalg.norm(L, axis=-1)
    def jac(self, X):
        L, c = self.legs(X); u = L / np.linalg.norm(L, axis=-1, keepdims=True)
        return np.concatenate([u, np.cross(c, u)], -1)
    def retract(self, X, d):
        p = X[:, :3] + d[:, :3]
        R = expso3(d[:, 3:]) @ self.R(X)
        return np.concatenate([p, R2rpy(R)], 1)
    def pose_error(self, Xh, X):
        ep = np.linalg.norm(Xh[:, :3] - X[:, :3], axis=1)
        Rr = np.einsum('nji,njk->nik', self.R(Xh), self.R(X))
        er = np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
        return ep, er
    def sample(self, rng, M, region='id'):
        b = self.box
        xy = rng.uniform(-b['xy'], b['xy'], (M, 2)); z = rng.uniform(*b['z'], M)
        yaw = np.deg2rad(rng.uniform(-b['yaw_deg'], b['yaw_deg'], M))
        if region == 'id':
            tilt = np.deg2rad(rng.uniform(-b['tilt_deg'], b['tilt_deg'], (M, 2)))
        else:
            lo, hi = self.shift_band_deg
            tilt = rng.uniform(-hi, hi, (M, 2))
            keep = np.max(np.abs(tilt), axis=1) > lo          # at least one tilt angle in the band
            tilt = np.deg2rad(np.where(keep[:, None], tilt, np.nan))
        X = np.column_stack([xy, z, tilt, yaw])
        return X[np.all(np.isfinite(X), axis=1)]
    def in_box(self, X):
        b = self.box
        return ((np.abs(X[:, 0]) <= b['xy']) & (np.abs(X[:, 1]) <= b['xy']) & (X[:, 2] >= b['z'][0]) &
                (X[:, 2] <= b['z'][1]) & (np.max(np.abs(X[:, 3:5]), axis=1) <= np.deg2rad(b['tilt_deg'])) &
                (np.abs(X[:, 5]) <= np.deg2rad(b['yaw_deg'])))

def lci(mech, X):
    """Local conditioning index 1/kappa of the dimensionally homogenised IK Jacobian, and det J."""
    J = mech.jac(X); Jh = J.copy(); Jh[..., mech.rot] = Jh[..., mech.rot] / mech.Lc
    sv = np.linalg.svd(Jh, compute_uv=False)
    return sv[:, -1] / sv[:, 0], np.linalg.det(J)

def sample_region(mech, rng, N, region='id'):
    """Rejection sampling: fixed working mode (sign of det J) and conditioning floor."""
    out, got = [], 0
    while got < N:
        X = mech.sample(rng, 3 * N, region)
        l, d = lci(mech, X)
        X = X[(l >= CFG['lci_min']) & (np.sign(d) == mech.aspect)]
        out.append(X); got += len(X)
    return np.concatenate(out)[:N]

MECHS = [Planar3RPR(), GoughStewart66()]

# %% [markdown]
# ### Jacobian verification (central finite differences on the twist coordinates)

# %%
rng_chk = np.random.default_rng(CFG['seed'])
jac_check = []
for m in MECHS:
    X = sample_region(m, rng_chk, 200)
    J = m.jac(X); Jn = np.zeros_like(J); h = 1e-7
    for j in range(m.n):
        d = np.zeros((len(X), m.n)); d[:, j] = h
        Jn[:, :, j] = (m.ik(m.retract(X, d)) - m.ik(m.retract(X, -d))) / (2 * h)
    err = float(np.abs(J - Jn).max())
    assert err < 1e-6, f'Jacobian check failed for {m.name}: {err}'
    jac_check.append({'mechanism': m.name, 'max_abs_jacobian_error': err})
savetab(pd.DataFrame(jac_check), 'T0_jacobian_check')

# %% [markdown]
# ## 2. Synthetic data
# Poses are sampled uniformly in the workspace box, restricted to one working mode and to LCI >= 0.01.
# Measured actuator lengths are $\tilde{\mathbf q}=\mathbf g(\mathbf x)+\boldsymbol\nu$, $\boldsymbol\nu\sim\mathcal N(\mathbf 0,\sigma^2\mathbf I)$.

# %%
DATA = {}
for mi, m in enumerate(MECHS):
    rng = np.random.default_rng(CFG['seed'] + 100 * (mi + 1))
    N = CFG['n_train'] + CFG['n_fit'] + CFG['n_pool']
    X = sample_region(m, rng, N, 'id')
    Q = m.ik(X) + rng.normal(0.0, CFG['sigma_q'], (N, m.n))
    Xs = sample_region(m, rng, CFG['n_shift'], 'shift')
    Qs = m.ik(Xs) + rng.normal(0.0, CFG['sigma_q'], (len(Xs), m.n))
    i1, i2 = CFG['n_train'], CFG['n_train'] + CFG['n_fit']
    D = {'X': X, 'Q': Q, 'tr': np.arange(0, i1), 'fit': np.arange(i1, i2), 'pool': np.arange(i2, N),
         'Xs': Xs, 'Qs': Qs}
    D['lci'], D['det'] = lci(m, X); D['lci_s'], _ = lci(m, Xs)
    DATA[m.key] = D
rows = []
for m in MECHS:
    D = DATA[m.key]
    rows.append({'mechanism': m.name, 'n_total_id': len(D['X']), 'n_shift': len(D['Xs']),
                 'q_min_m': D['Q'].min(), 'q_max_m': D['Q'].max(),
                 'LCI_min': D['lci'].min(), 'LCI_p10': np.percentile(D['lci'], 10),
                 'LCI_median': np.median(D['lci']), 'LCI_max': D['lci'].max(),
                 'LCI_shift_median': np.median(D['lci_s'])})
savetab(pd.DataFrame(rows).round(4), 'T1_dataset_summary')

# %% [markdown]
# ## 3. Learned forward kinematics (MLP) and deep-ensemble baseline

# %%
class LearnedFK:
    def __init__(self, Qtr, Xtr, n_ens, seed):
        self.sq = StandardScaler().fit(Qtr); self.sx = StandardScaler().fit(Xtr)
        self.nets, self.fit_times, self.epochs = [], [], []
        for i in range(n_ens):
            net = MLPRegressor(hidden_layer_sizes=CFG['hidden'], activation='relu', solver='adam',
                               learning_rate_init=2e-3, batch_size=256, max_iter=CFG['max_epochs'],
                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=12,
                               tol=1e-6, random_state=seed + i)
            t = time.perf_counter()
            net.fit(self.sq.transform(Qtr), self.sx.transform(Xtr))
            self.fit_times.append(time.perf_counter() - t); self.epochs.append(net.n_iter_)
            self.nets.append(net)
    def predict(self, Q, i=0):
        return self.sx.inverse_transform(self.nets[i].predict(self.sq.transform(Q)))
    def predict_all(self, Q):
        return np.stack([self.predict(Q, i) for i in range(len(self.nets))])

MODELS, train_rows = {}, []
for mi, m in enumerate(MECHS):
    D = DATA[m.key]
    fk = LearnedFK(D['Q'][D['tr']], D['X'][D['tr']], CFG['n_ens'], CFG['seed'] + 10 * mi)
    MODELS[m.key] = fk
    train_rows.append({'mechanism': m.name, 'members': CFG['n_ens'], 'hidden': str(CFG['hidden']),
                       'epochs_member0': fk.epochs[0], 'epochs_mean': np.mean(fk.epochs),
                       'fit_time_member0_s': fk.fit_times[0], 'fit_time_total_s': np.sum(fk.fit_times)})
    log(f'{m.name}: trained {CFG["n_ens"]} MLPs in {np.sum(fk.fit_times):.1f} s')
savetab(pd.DataFrame(train_rows).round(2), 'T2_training_summary')

# %% [markdown]
# ## 4. IK-consistency residual, Newton refinement and the IKC error estimator
# At a delivered pose $\hat{\mathbf x}$: $\mathbf r=\tilde{\mathbf q}-\mathbf g(\hat{\mathbf x})$,
# $\boldsymbol\delta=\mathbf J(\hat{\mathbf x})^{-1}\mathbf r$, $\mathbf C=(\mathbf J^{\mathsf T}\mathbf J)^{-1}$ and
# $\hat e_P=\sqrt{\lVert\mathbf P\boldsymbol\delta\rVert^2+\sigma^2\,\mathrm{tr}(\mathbf P\mathbf C\mathbf P^{\mathsf T})}$
# for the position block or the rotation block $\mathbf P$. Stage S$k$ = learned FK followed by $k$ Newton steps.

# %%
def batched_solve(J, r):
    try:
        return np.linalg.solve(J, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return (np.linalg.pinv(J) @ r[..., None])[..., 0]

def ikc_estimate(m, X, Q, sigma):
    r = Q - m.ik(X); J = m.jac(X)
    d = batched_solve(J, r)
    C = np.linalg.pinv(np.einsum('nji,njk->nik', J, J))
    res_p = np.linalg.norm(d[:, m.pos], axis=1); res_r = np.linalg.norm(d[:, m.rot], axis=1)
    noi_p = sigma * np.sqrt(np.clip(np.trace(C[:, m.pos, m.pos], axis1=1, axis2=2), 0, None))
    noi_r = sigma * np.sqrt(np.clip(np.trace(C[:, m.rot, m.rot], axis1=1, axis2=2), 0, None))
    return {'delta': d, 'res_p': res_p, 'res_r': res_r, 'noi_p': noi_p, 'noi_r': noi_r,
            'hat_p': np.sqrt(res_p**2 + noi_p**2), 'hat_r': np.sqrt(res_r**2 + noi_r**2),
            'resid_norm': np.linalg.norm(r, axis=1)}

def run_stages(m, fk, Q, K=2):
    Xs, Es = [fk.predict(Q, 0)], []
    for k in range(K + 1):
        E = ikc_estimate(m, Xs[k], Q, CFG['sigma_q']); Es.append(E)
        if k < K:
            Xs.append(m.retract(Xs[k], E['delta']))
    return Xs, Es

def newton_solve(m, X0, Q, max_iter, tol):
    X = X0.copy(); N = len(X)
    iters = np.full(N, max_iter); done = np.zeros(N, bool); alive = np.ones(N, bool)
    for k in range(max_iter + 1):
        r = Q - m.ik(X)
        ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(r), axis=1)
        alive &= ok
        new = alive & ~done & (np.abs(r).max(1) < tol)
        iters[new] = k; done |= new
        act = alive & ~done
        if k == max_iter or not act.any():
            break
        with np.errstate(all='ignore'):
            d = batched_solve(m.jac(X[act]), r[act])
            Xn = m.retract(X[act], d)
        bad = ~np.all(np.isfinite(Xn), axis=1) | (np.abs(Xn[:, :m.pos.stop]).max(1) > 10.0)
        Xn[bad] = X[act][bad]
        idx = np.where(act)[0]; alive[idx[bad]] = False
        X[act] = Xn
    return X, iters, done & alive

STAGE = {}
for m in MECHS:
    D, fk = DATA[m.key], MODELS[m.key]
    sel = np.concatenate([D['fit'], D['pool']])
    out = {}
    for tag, Q, Xt in [('id', D['Q'][sel], D['X'][sel]), ('shift', D['Qs'], D['Xs'])]:
        Xh, Es = run_stages(m, fk, Q, K=2)
        errs = [m.pose_error(x, Xt) for x in Xh]
        P = fk.predict_all(Q)
        sd = P.std(0)
        out[tag] = {'Xh': Xh, 'E': Es, 'e_p': [e[0] for e in errs], 'e_r': [e[1] for e in errs],
                    'ens_p': np.linalg.norm(sd[:, m.pos], axis=1), 'ens_r': np.linalg.norm(sd[:, m.rot], axis=1),
                    'lci_hat': [lci(m, x)[0] for x in Xh]}
    out['n_fit'] = len(D['fit'])
    STAGE[m.key] = out
log('Stage predictions and IKC estimates computed.')

# %% [markdown]
# ## 5. Point accuracy, classical Newton baseline and the consistency-versus-correctness analysis

# %%
acc_rows, branch_rows, BRANCH = [], [], {}
for m in MECHS:
    D, S = DATA[m.key], STAGE[m.key]
    nf = S['n_fit']; ip = slice(nf, None)
    Xt = D['X'][D['pool']]; Qt = D['Q'][D['pool']]
    for k in range(3):
        ep, er = S['id']['e_p'][k][ip], S['id']['e_r'][k][ip]
        acc_rows.append({'mechanism': m.name, 'solver': f'S{k}: learned FK + {k} Newton step(s)',
                         'pos_med_um': np.median(ep)*1e6, 'pos_p95_um': np.percentile(ep, 95)*1e6,
                         'pos_max_um': ep.max()*1e6, 'rot_med_deg': np.degrees(np.median(er)),
                         'rot_p95_deg': np.degrees(np.percentile(er, 95)), 'rot_max_deg': np.degrees(er.max()),
                         'converged_pct': np.nan, 'iters_mean': k})
    # noise-propagated RMS at the true pose (used to separate noise from branch errors)
    Jt = m.jac(Xt); Ct = np.linalg.pinv(np.einsum('nji,njk->nik', Jt, Jt))
    nrms_p = CFG['sigma_q'] * np.sqrt(np.trace(Ct[:, m.pos, m.pos], axis1=1, axis2=2))
    nrms_r = CFG['sigma_q'] * np.sqrt(np.trace(Ct[:, m.rot, m.rot], axis1=1, axis2=2))
    for label, X0, iters_max in [('Newton from home pose', np.repeat(m.home[None], len(Qt), 0), CFG['home_max_iter']),
                                 ('Learned FK + Newton to convergence', S['id']['Xh'][0][ip], CFG['converge_steps'])]:
        Xc, it, conv = newton_solve(m, X0, Qt, iters_max, CFG['newton_tol'])
        ep, er = m.pose_error(Xc, Xt)
        wrong = conv & ((ep > np.maximum(1e-3, 10 * nrms_p)) | (er > np.maximum(np.deg2rad(0.5), 10 * nrms_r)))
        _, dc = lci(m, Xc)
        lc = np.full(len(Xc), np.nan); lc[conv] = lci(m, Xc[conv])[0]
        aspect_flag = np.sign(dc) != m.aspect
        domain_flag = ~m.in_box(Xc) | ~(lc >= CFG['lci_min'])
        caught = wrong & (aspect_flag | domain_flag)
        ok = conv & ~wrong
        acc_rows.append({'mechanism': m.name, 'solver': label,
                         'pos_med_um': np.median(ep[ok])*1e6, 'pos_p95_um': np.percentile(ep[ok], 95)*1e6,
                         'pos_max_um': ep[ok].max()*1e6, 'rot_med_deg': np.degrees(np.median(er[ok])),
                         'rot_p95_deg': np.degrees(np.percentile(er[ok], 95)), 'rot_max_deg': np.degrees(er[ok].max()),
                         'converged_pct': 100*conv.mean(), 'iters_mean': it[conv].mean()})
        branch_rows.append({'mechanism': m.name, 'solver': label, 'n_test': len(Qt),
                            'not_converged': int((~conv).sum()), 'converged_wrong_solution': int(wrong.sum()),
                            'wrong_flagged_by_aspect_guard': int((wrong & aspect_flag).sum()),
                            'wrong_flagged_by_domain_guard': int((wrong & domain_flag).sum()),
                            'wrong_flagged_by_either': int(caught.sum()),
                            'silent_after_guards': int((wrong & ~caught).sum()),
                            'false_alarms_on_correct': int((ok & (aspect_flag | domain_flag)).sum()),
                            'wrong_true_LCI_max': float(D['lci'][D['pool']][wrong].max()) if wrong.any() else np.nan})
        BRANCH[(m.key, label)] = {'wrong': wrong, 'caught': caught, 'Xt': Xt, 'Xc': Xc, 'conv': conv}
savetab(pd.DataFrame(acc_rows).round(4), 'T3_point_accuracy')
savetab(pd.DataFrame(branch_rows), 'T4_branch_analysis')

# %% [markdown]
# ## 6. Learned-scale baseline (locally weighted conformal)
# A small MLP regresses log error on the measured actuator lengths using the separate fit split.

# %%
SCALES = {}
for mi, m in enumerate(MECHS):
    D, S, fk = DATA[m.key], STAGE[m.key], MODELS[m.key]
    nf = S['n_fit']; Qfit = fk.sq.transform(D['Q'][D['fit']])
    SC = {}
    for k in range(3):
        for t in ('p', 'r'):
            e = S['id'][f'e_{t}'][k]
            reg = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', learning_rate_init=2e-3,
                               batch_size=256, max_iter=CFG['max_epochs'], early_stopping=True,
                               n_iter_no_change=12, random_state=CFG['seed'] + 7 * k + mi)
            reg.fit(Qfit, np.log(e[:nf] + 1e-12))
            SC[('learn', k, t, 'id')] = np.exp(reg.predict(fk.sq.transform(D['Q'][D['pool']])))
            SC[('learn', k, t, 'shift')] = np.exp(reg.predict(fk.sq.transform(D['Qs'])))
    SCALES[m.key] = SC
log('Learned-scale baselines trained.')

# %% [markdown]
# ## 7. Split conformal prediction: six scale choices and conditional coverage
# Score $s=e/\hat s(\tilde{\mathbf q})$, quantile $\hat Q=s_{(\lceil (n+1)(1-\alpha)\rceil)}$, bound $U=\hat Q\,\hat s$.
# CP-strat and CP-IKC-M are group-conditional (Mondrian) over quintiles of the LCI evaluated at the delivered pose.
# Evaluation strata are deciles of the LCI at the true pose.

# %%
def conformal_q(s, alpha):
    n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.partition(s, k - 1)[k - 1]

def get_arrays(m, k, t, region):
    """Errors, scales and stratification variables on the pool (id) or the shift set."""
    S, SC = STAGE[m.key], SCALES[m.key]
    nf = S['n_fit']; sl = slice(nf, None) if region == 'id' else slice(None)
    R = S['id'] if region == 'id' else S['shift']
    e = R[f'e_{t}'][k][sl]
    sc = {'const': np.ones_like(e), 'ens': R[f'ens_{t}'][sl], 'learn': SC[('learn', k, t, region)],
          'ikc': R['E'][k][f'hat_{t}'][sl]}
    return e, sc, R['lci_hat'][k][sl]

def bounds_for(method, e_c, sc_c, l_c, sc_t, l_t, alpha, nb):
    if method in ('const', 'ens', 'learn', 'ikc'):
        return conformal_q(e_c / sc_c[method], alpha) * sc_t[method]
    base = 'const' if method == 'strat' else 'ikc'
    edges = np.quantile(l_c, np.linspace(0, 1, nb + 1)[1:-1])
    bc, bt = np.digitize(l_c, edges), np.digitize(l_t, edges)
    U = np.empty(len(l_t))
    for b in range(nb):
        mc = bc == b
        U[bt == b] = conformal_q(e_c[mc] / sc_c[base][mc], alpha) * sc_t[base][bt == b]
    return U

def evaluate(e_t, U, bins, nbins=10):
    cov = e_t <= U; cnt = np.bincount(bins, minlength=nbins)
    dcov = np.bincount(bins, weights=cov, minlength=nbins) / np.maximum(cnt, 1)
    dwid = np.bincount(bins, weights=U, minlength=nbins) / np.maximum(cnt, 1)
    miss = ~cov
    excess = 100 * (np.median(e_t[miss] / U[miss]) - 1) if miss.any() else 0.0
    return {'cov': cov.mean(), 'worst': dcov.min(), 'dcov': dcov, 'dwid': dwid, 'mean_w': U.mean(),
            'med_w': np.median(U), 'p90_w': np.percentile(U, 90), 'severe': np.mean(e_t > 2 * U),
            'excess': excess, 'cnt': cnt}

DEC_EDGES = {m.key: np.quantile(DATA[m.key]['lci'][DATA[m.key]['pool']], np.linspace(0, 1, 11)[1:-1]) for m in MECHS}
def true_bins(m, region='id'):
    l = DATA[m.key]['lci'][DATA[m.key]['pool']] if region == 'id' else DATA[m.key]['lci_s']
    return np.digitize(l, DEC_EDGES[m.key])

MAIN, DEC = [], {}
alpha, nb = CFG['alpha'], CFG['n_bins_mondrian']
for mi, m in enumerate(MECHS):
    rng = np.random.default_rng(CFG['seed'] + 500 + mi)
    bins_all = true_bins(m)
    perms = [rng.permutation(CFG['n_pool']) for _ in range(CFG['n_reps'])]
    for k in range(3):
        for t in ('p', 'r'):
            e, sc, l = get_arrays(m, k, t, 'id')
            acc = {mt: [] for mt in METHODS}
            for p in perms:
                c, te = p[:CFG['n_cal']], p[CFG['n_cal']:]
                sc_c = {kk: v[c] for kk, v in sc.items()}; sc_t = {kk: v[te] for kk, v in sc.items()}
                for mt in METHODS:
                    U = bounds_for(mt, e[c], sc_c, l[c], sc_t, l[te], alpha, nb)
                    acc[mt].append(evaluate(e[te], U, bins_all[te]))
            for mt in METHODS:
                A = acc[mt]
                MAIN.append({'mechanism': m.name, 'stage': f'S{k}', 'error': 'position' if t == 'p' else 'orientation',
                             'method': MNAME[mt],
                             'coverage_mean': np.mean([a['cov'] for a in A]), 'coverage_sd': np.std([a['cov'] for a in A]),
                             'worst_decile_cov': np.mean([a['worst'] for a in A]),
                             'lowest_LCI_decile_cov': np.mean([a['dcov'][0] for a in A]),
                             'highest_LCI_decile_cov': np.mean([a['dcov'][-1] for a in A]),
                             'mean_width': np.mean([a['mean_w'] for a in A]), 'median_width': np.mean([a['med_w'] for a in A]),
                             'p90_width': np.mean([a['p90_w'] for a in A]),
                             'severe_miss_pct': 100 * np.mean([a['severe'] for a in A]),
                             'median_miss_excess_pct': np.mean([a['excess'] for a in A])})
                DEC[(m.key, k, t, mt)] = (np.mean([a['dcov'] for a in A], 0), np.mean([a['dwid'] for a in A], 0))
main_df = pd.DataFrame(MAIN)
# units: position widths in micrometres, orientation widths in degrees
main_df['width_unit'] = np.where(main_df['error'] == 'position', 'um', 'deg')
fac = np.where(main_df['error'] == 'position', 1e6, 180 / np.pi)
main_df['mean_width'] *= fac; main_df['median_width'] *= fac; main_df['p90_width'] *= fac
savetab(main_df.round(4), 'T5_conformal_main')

# %% [markdown]
# ### Per-decile error and per-decile coverage/width (values quoted in the text)

# %%
dec_rows, dcw_rows = [], []
for m in MECHS:
    S = STAGE[m.key]; nf = S['n_fit']; bins_all = true_bins(m)
    for k in range(3):
        e = S['id']['e_p'][k][nf:] * 1e6
        for b in range(10):
            dec_rows.append({'mechanism': m.name, 'stage': f'S{k}', 'LCI_decile': b + 1,
                             'pos_median_um': np.median(e[bins_all == b]), 'pos_p90_um': np.percentile(e[bins_all == b], 90)})
    for mt in METHODS:
        dc, dw = DEC[(m.key, 1, 'p', mt)]
        for b in range(10):
            dcw_rows.append({'mechanism': m.name, 'stage': 'S1', 'method': MNAME[mt], 'LCI_decile': b + 1,
                             'coverage': dc[b], 'mean_width_um': dw[b] * 1e6})
savetab(pd.DataFrame(dec_rows).round(3), 'T12_error_by_decile')
savetab(pd.DataFrame(dcw_rows).round(4), 'T13_decile_coverage_width_S1_position')

# %% [markdown]
# ### Ranking quality of each scale and approximate pivotality of the IKC score
# If the IKC estimator is a first-order conditional RMS error, $E[s^2]=E[e^2/\hat e^2]\approx 1$ in every conditioning decile.

# %%
rank_rows, piv_rows = [], []
for m in MECHS:
    bins_all = true_bins(m)
    for k in range(3):
        for t in ('p', 'r'):
            e, sc, _ = get_arrays(m, k, t, 'id')
            row = {'mechanism': m.name, 'stage': f'S{k}', 'error': 'position' if t == 'p' else 'orientation'}
            for mt in ('ens', 'learn', 'ikc'):
                row[f'spearman_{mt}'] = spearmanr(sc[mt], e)[0]
            rank_rows.append(row)
            s2 = (e / sc['ikc'])**2
            ds2 = [np.mean(s2[bins_all == b]) for b in range(10)]
            piv_rows.append({'mechanism': m.name, 'stage': f'S{k}', 'error': row['error'],
                             'E_s2_all': np.mean(s2), 'E_s2_lowest_LCI_decile': ds2[0],
                             'E_s2_highest_LCI_decile': ds2[-1], 'E_s2_min_decile': np.min(ds2),
                             'E_s2_max_decile': np.max(ds2), 'median_s_all': np.median(np.sqrt(s2))})
savetab(pd.DataFrame(rank_rows).round(3), 'T6_rank_correlation')
savetab(pd.DataFrame(piv_rows).round(3), 'T7_pivotality')

# %% [markdown]
# ## 8. Distribution shift
# **SH1 (conditioning shift):** calibration drawn only from the better-conditioned half of the pool
# (true LCI above the pool median); test on all conditioning levels.
# **SH2 (workspace extrapolation):** model and calibration from the nominal workspace; test poses from an
# orientation band outside it (Gough-Stewart: tilt 25-30 deg; 3-RPR: |phi| 50-60 deg).

# %%
SHIFT = []
k_sh = 1
for mi, m in enumerate(MECHS):
    rng = np.random.default_rng(CFG['seed'] + 900 + mi)
    lpool = DATA[m.key]['lci'][DATA[m.key]['pool']]
    good = np.where(lpool >= np.median(lpool))[0]
    bins_all = true_bins(m); bins_sh = true_bins(m, 'shift')
    for t in ('p', 'r'):
        e, sc, l = get_arrays(m, k_sh, t, 'id')
        es, scs, ls = get_arrays(m, k_sh, t, 'shift')
        for scen in ('SH1', 'SH2'):
            acc = {mt: [] for mt in METHODS}
            for r_ in range(CFG['n_reps']):
                if scen == 'SH1':
                    c = rng.choice(good, CFG['n_cal'], replace=False)
                    te = np.setdiff1d(np.arange(CFG['n_pool']), c)
                    et, sct, lt, bt = e[te], {kk: v[te] for kk, v in sc.items()}, l[te], bins_all[te]
                else:
                    c = rng.choice(CFG['n_pool'], CFG['n_cal'], replace=False)
                    et, sct, lt, bt = es, scs, ls, bins_sh
                sc_c = {kk: v[c] for kk, v in sc.items()}
                for mt in METHODS:
                    U = bounds_for(mt, e[c], sc_c, l[c], sct, lt, alpha, nb)
                    acc[mt].append(evaluate(et, U, bt))
            for mt in METHODS:
                A = acc[mt]
                SHIFT.append({'mechanism': m.name, 'scenario': scen, 'stage': f'S{k_sh}',
                              'error': 'position' if t == 'p' else 'orientation', 'method': MNAME[mt],
                              'coverage_mean': np.mean([a['cov'] for a in A]), 'coverage_sd': np.std([a['cov'] for a in A]),
                              'worst_decile_cov': np.mean([np.min(a['dcov'][a['cnt'] >= 50]) for a in A]),
                              'deciles_with_50plus': int(np.sum(A[0]['cnt'] >= 50)),
                              'severe_miss_pct': 100 * np.mean([a['severe'] for a in A])})
savetab(pd.DataFrame(SHIFT).round(4), 'T8_shift')

# %% [markdown]
# ## 9. Ablations of the IKC scale and sensitivity to the calibration size

# %%
ABL = []
for mi, m in enumerate(MECHS):
    rng = np.random.default_rng(CFG['seed'] + 1300 + mi)
    S = STAGE[m.key]; nf = S['n_fit']; bins_all = true_bins(m)
    perms = [rng.permutation(CFG['n_pool']) for _ in range(CFG['n_reps'])]
    for k in (1, 2):
        E = S['id']['E'][k]
        for t in ('p', 'r'):
            e = S['id'][f'e_{t}'][k][nf:]
            res, noi = E[f'res_{t}'][nf:], E[f'noi_{t}'][nf:]
            variants = {'IKC (residual + noise)': np.sqrt(res**2 + noi**2),
                        'residual term only': res + 1e-15, 'noise term only': noi,
                        'noise term with 0.5 sigma': np.sqrt(res**2 + (0.5 * noi)**2),
                        'noise term with 2 sigma': np.sqrt(res**2 + (2.0 * noi)**2)}
            for vn, s in variants.items():
                A = []
                for p in perms:
                    c, te = p[:CFG['n_cal']], p[CFG['n_cal']:]
                    U = conformal_q(e[c] / s[c], alpha) * s[te]
                    A.append(evaluate(e[te], U, bins_all[te]))
                w = 1e6 if t == 'p' else 180 / np.pi
                ABL.append({'mechanism': m.name, 'stage': f'S{k}', 'error': 'position' if t == 'p' else 'orientation',
                            'variant': vn, 'coverage_mean': np.mean([a['cov'] for a in A]),
                            'worst_decile_cov': np.mean([a['worst'] for a in A]),
                            'mean_width': w * np.mean([a['mean_w'] for a in A])})
savetab(pd.DataFrame(ABL).round(4), 'T9_ablation')

NCAL = []
for mi, m in enumerate(MECHS):
    rng = np.random.default_rng(CFG['seed'] + 1700 + mi)
    e, sc, l = get_arrays(m, 1, 'p', 'id')
    for n in CFG['ncal_grid']:
        covs = {mt: [] for mt in ('const', 'ikc')}
        for r_ in range(CFG['ncal_reps']):
            p = rng.permutation(CFG['n_pool']); c, te = p[:n], p[CFG['n_cal']:]
            for mt in covs:
                U = conformal_q(e[c] / sc[mt][c], alpha) * sc[mt][te]
                covs[mt].append(np.mean(e[te] <= U))
        for mt, v in covs.items():
            NCAL.append({'mechanism': m.name, 'n_cal': n, 'method': MNAME[mt], 'coverage_mean': np.mean(v),
                         'coverage_sd': np.std(v), 'coverage_p05': np.percentile(v, 5), 'coverage_p95': np.percentile(v, 95)})
savetab(pd.DataFrame(NCAL).round(4), 'T10_ncal_sensitivity')

# %% [markdown]
# ## 10. Computational cost (single query and batched)

# %%
def mlp_forward_numpy(fk, q):
    z = (q - fk.sq.mean_) / fk.sq.scale_
    net = fk.nets[0]
    for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        z = z @ W + b
        if i < len(net.coefs_) - 1:
            z = np.maximum(z, 0.0)
    return z * fk.sx.scale_ + fk.sx.mean_

TIME = []
for m in MECHS:
    D, fk = DATA[m.key], MODELS[m.key]
    Qp = D['Q'][D['pool']]; nq = CFG['timing_queries']
    assert np.allclose(mlp_forward_numpy(fk, Qp[:50]), fk.predict(Qp[:50]), atol=1e-9)
    def one_query(i):
        q = Qp[i:i+1]
        x0 = mlp_forward_numpy(fk, q)
        E0 = ikc_estimate(m, x0, q, CFG['sigma_q'])
        x1 = m.retract(x0, E0['delta'])
        E1 = ikc_estimate(m, x1, q, CFG['sigma_q'])
        return x1, E1['hat_p'], E1['hat_r']
    t = time.perf_counter()
    for i in range(nq): mlp_forward_numpy(fk, Qp[i:i+1])
    t_mlp = (time.perf_counter() - t) / nq
    t = time.perf_counter()
    for i in range(nq): one_query(i)
    t_full = (time.perf_counter() - t) / nq
    Xp = D['X'][D['pool']]
    t = time.perf_counter()
    for i in range(nq): ikc_estimate(m, Xp[i:i+1], Qp[i:i+1], CFG['sigma_q'])
    t_ikc = (time.perf_counter() - t) / nq
    t = time.perf_counter()
    for i in range(nq): fk.predict_all(Qp[i:i+1])
    t_ens = (time.perf_counter() - t) / nq
    nb_ = min(20000, len(Qp))
    Xh, Es = run_stages(m, fk, Qp[:nb_], K=1)          # warm-up
    t = time.perf_counter(); Xh, Es = run_stages(m, fk, Qp[:nb_], K=1); t_batch = (time.perf_counter() - t) / nb_
    TIME.append({'mechanism': m.name, 'mlp_forward_us': 1e6 * t_mlp, 'one_ikc_evaluation_us': 1e6 * t_ikc,
                 'S1_certified_query_us': 1e6 * t_full, 'ensemble_5_sklearn_predict_us': 1e6 * t_ens,
                 'S1_certified_batched_us_per_pose': 1e6 * t_batch})
savetab(pd.DataFrame(TIME).round(2), 'T11_timing')

# %% [markdown]
# ## 11. Figures (one plot per figure)

# %%
def fig_rpr_schematic():
    m = MECHS[0]; X = np.array([[0.52, 0.36, np.deg2rad(12)]])
    L, c = m.legs(X); P = X[0, :2] + c[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    b = m.box
    ax.add_patch(plt.Rectangle((b['x'][0], b['y'][0]), b['x'][1]-b['x'][0], b['y'][1]-b['y'][0], fill=False,
                               ls='--', lw=1.0, ec='#555555', label='workspace box (x, y)'))
    ax.fill(*np.vstack([m.A, m.A[:1]]).T, color='#dddddd', alpha=0.6, lw=0)
    for i in range(3):
        ax.plot([m.A[i, 0], P[i, 0]], [m.A[i, 1], P[i, 1]], color='#1f3b73', lw=2.2)
        ax.plot(*m.A[i], 'o', ms=8, mfc='white', mec='black', mew=1.4)
        ax.annotate(f'$A_{i+1}$', m.A[i], xytext=(-14, -14 if i < 2 else 6), textcoords='offset points')
        ax.annotate(f'$q_{i+1}$', (m.A[i] + P[i]) / 2, xytext=(5, 2), textcoords='offset points', color='#1f3b73')
    ax.fill(*np.vstack([P, P[:1]]).T, color='#d62728', alpha=0.35, ec='#8b0000', lw=1.5)
    for i in range(3):
        ax.plot(*P[i], 'o', ms=6, color='#8b0000')
        ax.annotate(f'$B_{i+1}$', P[i], xytext=(6, 4), textcoords='offset points', color='#8b0000')
    ax.plot(*X[0, :2], 'k+', ms=10, mew=1.5); ax.annotate(r'$(x, y, \varphi)$', X[0, :2], xytext=(8, -14), textcoords='offset points')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal'); ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 0.95)
    savefig(fig, 'F02_rpr_schematic')

def fig_gsp_schematic():
    m = MECHS[1]; X = np.array([[0.04, -0.03, 0.56, np.deg2rad(8), np.deg2rad(-6), np.deg2rad(20)]])
    L, c = m.legs(X); P = X[0, :3] + c[0]
    fig = plt.figure(figsize=(5.4, 4.6)); ax = fig.add_subplot(projection='3d')
    th = np.linspace(0, 2*np.pi, 100)
    ax.plot(m.Rb*np.cos(th), m.Rb*np.sin(th), 0*th, color='#999999', lw=0.8)
    ax.plot(*np.vstack([m.A, m.A[:1]]).T, color='black', lw=1.4)
    ax.plot(*np.vstack([P, P[:1]]).T, color='#8b0000', lw=1.8)
    ax.plot_trisurf(P[:, 0], P[:, 1], P[:, 2], color='#d62728', alpha=0.25)
    for i in range(6):
        ax.plot([m.A[i, 0], P[i, 0]], [m.A[i, 1], P[i, 1]], [m.A[i, 2], P[i, 2]], color='#1f3b73', lw=2.0)
        ax.scatter(*m.A[i], s=22, color='black'); ax.scatter(*P[i], s=16, color='#8b0000')
        ax.text(*(m.A[i] * 1.12), f'$A_{i+1}$', fontsize=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_box_aspect((1, 1, 0.65)); ax.view_init(elev=22, azim=-58)
    savefig(fig, 'F03_gsp_schematic')

def fig_rpr_lci_map():
    m = MECHS[0]
    xs, ys = np.linspace(0.0, 1.0, 321), np.linspace(-0.05, 0.85, 289)
    XX, YY = np.meshgrid(xs, ys); G = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    l, d = lci(m, G); l = l.reshape(XX.shape); d = d.reshape(XX.shape)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cf = ax.contourf(XX, YY, np.log10(np.maximum(l, 1e-4)), levels=np.linspace(-3, 0, 25), cmap='viridis')
    cb = fig.colorbar(cf, ax=ax); cb.set_label(r'$\log_{10}$ LCI at $\varphi = 0$')
    ax.contour(XX, YY, d, levels=[0.0], colors='white', linewidths=1.8)
    b = m.box
    ax.add_patch(plt.Rectangle((b['x'][0], b['y'][0]), b['x'][1]-b['x'][0], b['y'][1]-b['y'][0], fill=False, ls='--', lw=1.3, ec='white'))
    B = BRANCH[(m.key, 'Learned FK + Newton to convergence')]
    w = B['wrong']
    ax.scatter(B['Xt'][w, 0], B['Xt'][w, 1], s=28, marker='x', color='#ff4d4d', lw=1.4,
               label=f'true poses of silent wrong-solution cases (n = {int(w.sum())})')
    ax.plot([], [], color='white', lw=1.8, label=r'Type II singularity locus ($\det \mathbf{J}=0$) at $\varphi=0$')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.8, frameon=False)
    savefig(fig, 'F04_rpr_lci_map')

def fig_err_decile(m, name):
    S = STAGE[m.key]; nf = S['n_fit']; bins = true_bins(m)
    centers = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    styles = [('S0: learned FK', '#1f77b4', 'o'), ('S1: + 1 Newton step', '#d62728', 'D'), ('S2: + 2 Newton steps', '#2ca02c', 's')]
    for k, (lab, col, mk) in enumerate(styles):
        e = S['id']['e_p'][k][nf:] * 1e6
        med = [np.median(e[bins == b]) for b in range(10)]
        p90 = [np.percentile(e[bins == b], 90) for b in range(10)]
        ax.plot(centers, med, marker=mk, color=col, label=f'{lab} (median)')
        ax.plot(centers, p90, marker=mk, color=col, ls=':', mfc='none', label=f'{lab} (90th pct.)')
    ax.set_yscale('log'); ax.set_xticks(centers)
    ax.set_xlabel('LCI decile of the true pose (1 = nearest to Type II singularity)')
    ax.set_ylabel('position error [µm]')
    ax.legend(ncol=3, fontsize=7.0, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    savefig(fig, name)

def fig_scatter(m, k, name):
    S = STAGE[m.key]; nf = S['n_fit']
    e = S['id']['e_p'][k][nf:] * 1e6; h = S['id']['E'][k]['hat_p'][nf:] * 1e6
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    hb = ax.hexbin(h, e, xscale='log', yscale='log', gridsize=55, cmap='magma_r', mincnt=1, bins='log')
    lo, hi = min(h.min(), e.min()), max(h.max(), e.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, label=r'$e = \hat{e}$')
    cb = fig.colorbar(hb, ax=ax); cb.set_label(r'$\log_{10}$ count')
    ax.set_xlabel(r'IKC estimate $\hat{e}_P$ [µm] (no ground truth used)'); ax.set_ylabel(r'true position error $e_P$ [µm]')
    ax.legend(loc='upper left')
    savefig(fig, name)

def fig_cov_decile(m, k, t, name):
    fig, ax = plt.subplots(figsize=(5.6, 3.7)); centers = np.arange(1, 11)
    for mt in METHODS:
        dc, _ = DEC[(m.key, k, t, mt)]
        ax.plot(centers, dc, marker=MMARK[mt], color=MCOL[mt], label=MNAME[mt], lw=1.5 if 'ikc' not in mt else 2.2)
    ax.axhline(1 - CFG['alpha'], color='black', ls='--', lw=1.0, label='target 0.90')
    lo = min(min(DEC[(m.key, k, t, mt)][0].min() for mt in METHODS), 0.48)
    ax.set_xticks(centers); ax.set_ylim(np.floor((lo - 0.03) * 20) / 20, 1.01)
    ax.set_xlabel('LCI decile of the true pose (1 = nearest to Type II singularity)')
    ax.set_ylabel('empirical coverage')
    ax.legend(ncol=4, fontsize=7.0, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    savefig(fig, name)

def fig_width_decile(m, k, t, name):
    fig, ax = plt.subplots(figsize=(5.6, 3.7)); centers = np.arange(1, 11)
    w = 1e6 if t == 'p' else 180 / np.pi
    for mt in METHODS:
        _, dw = DEC[(m.key, k, t, mt)]
        ax.plot(centers, dw * w, marker=MMARK[mt], color=MCOL[mt], label=MNAME[mt], lw=1.5 if 'ikc' not in mt else 2.2)
    ax.set_yscale('log'); ax.set_xticks(centers)
    ax.set_xlabel('LCI decile of the true pose (1 = nearest to Type II singularity)')
    ax.set_ylabel('mean bound width [µm]' if t == 'p' else 'mean bound width [deg]')
    ax.legend(ncol=3, fontsize=7.0, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    savefig(fig, name)

def fig_shift():
    df = pd.DataFrame(SHIFT); df = df[df['error'] == 'position']
    groups = [(MECHS[1].name, 'SH1'), (MECHS[0].name, 'SH1'), (MECHS[1].name, 'SH2'), (MECHS[0].name, 'SH2')]
    labels = ['SH1\nGough-Stewart', 'SH1\n3-RPR', 'SH2\nGough-Stewart', 'SH2\n3-RPR']
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); wbar = 0.13; x = np.arange(len(groups))
    for j, mt in enumerate(METHODS):
        vals = [df[(df['mechanism'] == g[0]) & (df['scenario'] == g[1]) & (df['method'] == MNAME[mt])]['coverage_mean'].values[0] for g in groups]
        ax.bar(x + (j - 2.5) * wbar, vals, wbar, color=MCOL[mt], label=MNAME[mt], edgecolor='black', lw=0.4)
    ax.axhline(1 - CFG['alpha'], color='black', ls='--', lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.4, 1.0); ax.set_ylabel('marginal coverage (stage S1, position)')
    ax.legend(ncol=3, fontsize=7.0, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
    savefig(fig, 'F11_shift_coverage')

def fig_ncal():
    df = pd.DataFrame(NCAL); fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for m, ls in [(MECHS[1], '-'), (MECHS[0], '--')]:
        for mt in ('const', 'ikc'):
            d = df[(df['mechanism'] == m.name) & (df['method'] == MNAME[mt])]
            ax.plot(d['n_cal'], d['coverage_sd'], ls=ls, marker=MMARK[mt], color=MCOL[mt],
                    label=f'{MNAME[mt]}, {"Gough-Stewart" if m.key == "gsp" else "3-RPR"}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('calibration set size $n$'); ax.set_ylabel('std. of test coverage over 50 draws')
    ax.legend(fontsize=7.4)
    savefig(fig, 'F12_ncal_sd')

fig_rpr_schematic(); fig_gsp_schematic(); fig_rpr_lci_map()
fig_err_decile(MECHS[1], 'F05_gsp_error_by_decile'); fig_err_decile(MECHS[0], 'F06_rpr_error_by_decile')
fig_scatter(MECHS[1], 1, 'F07_gsp_ikc_scatter_S1')
fig_cov_decile(MECHS[1], 1, 'p', 'F08_gsp_coverage_by_decile_S1'); fig_cov_decile(MECHS[0], 1, 'p', 'F09_rpr_coverage_by_decile_S1')
fig_width_decile(MECHS[1], 1, 'p', 'F10_gsp_width_by_decile_S1')
fig_shift(); fig_ncal()

# %% [markdown]
# ## 12. Summary and run record

# %%
summary = {'config': CFG, 'runtime_s': time.time() - T_START,
           'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
                        'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__}}
with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)
log(f'\nTotal runtime: {summary["runtime_s"]:.1f} s')
log('Outputs written to', OUT.resolve())
LOG.close()
