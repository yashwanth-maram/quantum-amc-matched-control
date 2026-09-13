"""
E3 -- RadioML 2018.01A validation.  Run as Kaggle notebook cells, in order.

STATUS OF RESULTS
  VALID    encoding scale probe (Sec. V-E of the paper):
             pi   0.309 +/- 0.017      pi/4  0.516 +/- 0.027
             pi/2 0.533 +/- 0.043      pi/8  0.511 +/- 0.030
           Step at the injectivity boundary, not a gradient in angle
           magnitude -> causal support for Proposition 1.

  VALID    frozen-gate sweep, test acc flat while train acc climbs:
             K' =  3   12   24   48   72
             te  .304 .246 .335 .336 .308
             tr  .450 .538 .750 .863 .900
           Capacity is not the binding constraint.

  INVALID  kappa on 2018.01A.  Three of four SNR bands were degenerate
           (deep band never left chance; mid/high overfit base classes to
           proto-loss 0.01 with novel accuracy at chance).  The reported
           kappa_quantum = 0.932 +/- 0.242 and kappa_linear = 1.055 are
           ratios of chance over chance.  DO NOT CITE.
           Note also that CV_ratio/CV_offset = 8.3x looked like a clean
           replication of the 8.6x in the paper; it is an artifact.  With
           degenerate accuracies, differences cluster near zero so
           CV_offset explodes on a near-zero denominator.  Eq. (12)
           presupposes non-degenerate accuracies.

  TO FIX   Cell 8 needs early stopping on a held-out BASE validation split
           instead of a fixed 6000 episodes, and bands
           {(-8,-2),(0,6),(8,16),(18,30)} for a wider accuracy spread.

ENVIRONMENT
  Kaggle notebook.  Add Input -> pinxau1000/radioml2018 (mounts read-only,
  no download).  Internet OFF and NO pip installs: PennyLane pins numpy<2,
  which downgrades the base image and breaks ~15 preinstalled packages.
  The VQC below is an exact statevector simulator in pure PyTorch, which
  removes the dependency and is faster at n = 6.

CLASS ORDER -- the single most dangerous assumption here.
  The ordering shipped in the original DeepSig release is PERMUTED
  (index 0 is '32PSK', not 'OOK').  Use classes-fixed.json from the Kaggle
  mirror.  Verified independently by envelope statistics at +30 dB:
  OOK has the highest zero-fraction (0.117) and FM/GMSK the lowest
  envelope CV (0.005 / 0.084), both as expected.
"""

# ---------------------------------------------------------------- Cell 1
import os, gc, json, numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats

assert np.__version__.startswith("2"), \
    f"numpy {np.__version__}: container polluted by a pip install, start a fresh notebook"

D  = "/kaggle/input/datasets/pinxau1000/radioml2018"
H5 = os.path.join(D, "GOLD_XYZ_OSC.0001_1024.hdf5")
N_MOD, N_SNR, FRAMES = 24, 26, 4096
SNRS = np.arange(-20, 32, 2)

CLASSES = json.load(open(os.path.join(D, "classes-fixed.json")))
assert CLASSES[0] == "OOK" and CLASSES[21] == "FM" and len(CLASSES) == 24, \
    "WRONG CLASS ORDER - using DeepSig original, not classes-fixed"

NOVEL     = ["16QAM", "64QAM", "8PSK", "AM-DSB-SC"]      # mirrors the 2016.10a holdout
NOVEL_IDX = [CLASSES.index(c) for c in NOVEL]
BASE_IDX  = [i for i in range(N_MOD) if i not in NOVEL_IDX]

# ---------------------------------------------------------------- Cell 2
# Contiguous block reads.  The file is modulation-major then SNR-major, so
# X[:10000] is a single class, and h5py fancy indexing needs sorted unique
# indices and is far slower than slicing.
N_PER_CELL = 256                                   # 24*26*256 = 159,744, ~1.3 GB
rng = np.random.default_rng(0)

Xs, ys, zs = [], [], []
with h5py.File(H5, "r") as f:
    Xd = f["X"]
    for m in range(N_MOD):
        for s in range(N_SNR):
            base = (m * N_SNR + s) * FRAMES
            off  = int(rng.integers(0, FRAMES - N_PER_CELL))
            blk  = Xd[base + off : base + off + N_PER_CELL]
            Xs.append(blk.transpose(0, 2, 1).astype(np.float32))   # -> (N,2,1024)
            ys.append(np.full(N_PER_CELL, m, np.int16))
            zs.append(np.full(N_PER_CELL, SNRS[s], np.int16))

X = np.concatenate(Xs); y = np.concatenate(ys); z = np.concatenate(zs)
del Xs, ys, zs; gc.collect()
X /= (np.linalg.norm(X.reshape(len(X), -1), axis=1)[:, None, None] + 1e-12)
np.savez("/kaggle/working/e3_cache.npz", X=X, y=y, z=z)   # Save Version now

# ---------------------------------------------------------------- Cell 3
DEV, D_EMB = ("cuda" if torch.cuda.is_available() else "cpu"), 6   # d = n = 6 INVARIANT

class Extractor(nn.Module):
    """~9,014 params, matched to the 9,478 of the 2016.10a extractor.
    BatchNorm and the temperature divisor are load-bearing: without them
    the fc output saturates tanh (83% of coords at |z|>0.95), gradients
    die, and every head reads a constant vector at exactly chance."""
    def __init__(self, d=D_EMB):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),  nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(4))
        self.fc   = nn.Linear(64 * 4, d)
        self.temp = nn.Parameter(torch.tensor(2.0))
    def forward(self, x):
        return torch.tanh(self.fc(self.conv(x).flatten(1)) / self.temp.clamp(min=0.5))

def train_extractor(X, y, base_idx, episodes=6000, n_way=5, k_shot=5, q_query=15,
                    seed=0, snr_mask=None):
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    model = Extractor().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    m = snr_mask if snr_mask is not None else np.ones(len(y), bool)
    pool = {c: np.where((y == c) & m)[0] for c in base_idx}
    pool = {c: v for c, v in pool.items() if len(v) >= k_shot + q_query}
    model.train()
    for ep in range(episodes):
        cls = g.choice(list(pool.keys()), n_way, replace=False)
        si, qi = [], []
        for c in cls:
            p = g.choice(pool[c], k_shot + q_query, replace=False)
            si.append(p[:k_shot]); qi.append(p[k_shot:])
        xs = torch.tensor(X[np.concatenate(si)], device=DEV)
        xq = torch.tensor(X[np.concatenate(qi)], device=DEV)
        yq = torch.arange(n_way, device=DEV).repeat_interleave(q_query)
        es, eq = model(xs).view(n_way, k_shot, -1).mean(1), model(xq)
        loss = F.cross_entropy(-torch.cdist(eq, es) ** 2, yq)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 300 == 0:
            with torch.no_grad():
                sat = (eq.abs() > 0.95).float().mean().item()
            # GATE: loss must fall below ln(5)=1.6094 and sat stay under ~0.5
            print(f"  ep {ep:4d} proto-loss {loss.item():.4f} sat {sat:.2f}")
    model.eval(); return model

@torch.no_grad()
def embed(model, X, bs=2048):
    return np.concatenate([model(torch.tensor(X[i:i+bs], device=DEV)).cpu().numpy()
                           for i in range(0, len(X), bs)])

# ---------------------------------------------------------------- Cell 4
# Exact statevector simulation, n = 6 -> 64-dim state.  Validated against
# the analytic identities of Proposition 1: encoding-only gives
# <Z_i> = cos(pi z_i) to 4.2e-07, and f(z+2) = f(z) to 4.2e-07.
CDTYPE, N_QUBITS, N_LAYERS = torch.complex64, 6, 4

def _ry(s_, th, w):
    s = s_.movedim(w + 1, -1); sh = (-1,) + (1,) * (s.dim() - 2)
    c = torch.cos(th / 2).reshape(sh).to(s.dtype); n = torch.sin(th / 2).reshape(sh).to(s.dtype)
    a0, a1 = s[..., 0], s[..., 1]
    return torch.stack([c * a0 - n * a1, n * a0 + c * a1], -1).movedim(-1, w + 1)

def _rz(s_, th, w):
    s = s_.movedim(w + 1, -1); sh = (-1,) + (1,) * (s.dim() - 2)
    t = th.reshape(sh).to(s.dtype)
    return torch.stack([torch.exp(-0.5j * t) * s[..., 0],
                        torch.exp(0.5j * t) * s[..., 1]], -1).movedim(-1, w + 1)

def _cnot(s_, c, t):
    s = s_.movedim(c + 1, 1); a0, a1 = s[:, 0], s[:, 1]
    return torch.stack([a0, a1.flip(t + 1 if t < c else t)], 1).movedim(1, c + 1)

def _ez(s_, w):
    p = s_.abs() ** 2; s = p.movedim(w + 1, -1).reshape(p.shape[0], -1, 2).sum(1)
    return s[:, 0] - s[:, 1]

def _ezz(s_, i, j):
    p = s_.abs() ** 2
    s = p.movedim(i + 1, -1).movedim(j + 1 if j < i else j, -2).reshape(p.shape[0], -1, 2, 2).sum(1)
    return s[:, 0, 0] - s[:, 0, 1] - s[:, 1, 0] + s[:, 1, 1]

def _ex(s_, w):
    s = s_.movedim(w + 1, -1).reshape(s_.shape[0], -1, 2)
    return 2 * (s[..., 0].conj() * s[..., 1]).sum(1).real

class QuantumHead(nn.Module):
    """Data re-uploading VQC.  scale=pi reproduces the paper; scale=pi/2
    confines the encoding to an injective half-period (Sec. V-E)."""
    def __init__(self, n_cls, scale=np.pi, readout="single", n=N_QUBITS, L=N_LAYERS):
        super().__init__()
        self.n, self.L, self.scale, self.readout = n, L, scale, readout
        self.weights = nn.Parameter(0.1 * torch.randn(L, n, 3))         # 3Ln = 72
        self.out = nn.Linear(n if readout == "single" else 2 * n + n * (n - 1) // 2, n_cls)
    def forward(self, zin):
        B, n = zin.shape[0], self.n
        st = torch.zeros(B, *([2] * n), dtype=CDTYPE, device=zin.device)
        st.reshape(B, -1)[:, 0] = 1.0
        for l in range(self.L):
            for i in range(n): st = _ry(st, self.scale * zin[:, i], i)
            for i in range(n):                                          # Rot = RZ RY RZ
                st = _rz(st, self.weights[l, i, 0].expand(1), i)
                st = _ry(st, self.weights[l, i, 1].expand(1), i)
                st = _rz(st, self.weights[l, i, 2].expand(1), i)
            for i in range(n): st = _cnot(st, i, (i + 1) % n)            # ring, range 1
        obs = [_ez(st, i) for i in range(n)]
        if self.readout != "single":                                    # 2n + nC2 = 27
            obs += [_ezz(st, i, j) for i in range(n) for j in range(i + 1, n)]
            obs += [_ex(st, i) for i in range(n)]
        return self.out(torch.stack(obs, -1))

def MLPHead(n_cls):
    return nn.Sequential(nn.Linear(D_EMB, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, 64),   nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, n_cls))
def LinearHead(n_cls): return nn.Linear(D_EMB, n_cls)

def fit_head(kind, Zs, ys_, Zq, yq, n_cls, seed=0, epochs=150, scale=np.pi):
    if kind == "1nn":
        return KNeighborsClassifier(1).fit(Zs, ys_).score(Zq, yq)
    xs = torch.tensor(Zs, dtype=torch.float32); ts = torch.tensor(ys_, dtype=torch.long)
    xq = torch.tensor(Zq, dtype=torch.float32); tq = torch.tensor(yq, dtype=torch.long)
    best_loss, best_acc = np.inf, 0.0
    for lr in (0.05, 0.01):
        torch.manual_seed(seed)
        h = (QuantumHead(n_cls, scale=scale) if kind == "quantum"
             else {"mlp": MLPHead, "linear": LinearHead}[kind](n_cls))
        opt = torch.optim.Adam(h.parameters(), lr=lr); h.train()
        for _ in range(epochs):
            loss = F.cross_entropy(h(xs), ts)
            opt.zero_grad(); loss.backward(); opt.step()
        h.eval()
        with torch.no_grad():
            tr  = F.cross_entropy(h(xs), ts).item()
            acc = (h(xq).argmax(1) == tq).float().mean().item()
        if tr < best_loss:                       # select on TRAIN loss: no test leak
            best_loss, best_acc = tr, acc
    return best_acc

# ---------------------------------------------------------------- Cell 5
def band_mask(lo, hi): return (z >= lo) & (z <= hi)

def novel_pools(mask):
    sup, qi = {}, []
    for new_c, old_c in enumerate(NOVEL_IDX):
        idx = np.where((y == old_c) & mask)[0]; cut = len(idx) // 2
        sup[new_c] = idx[:cut]; qi.append((idx[cut:], new_c))
    Xq = np.concatenate([X[i] for i, _ in qi])
    yq = np.concatenate([np.full(len(i), c) for i, c in qi])
    return sup, Xq, yq

ext = train_extractor(X, y, BASE_IDX, seed=0, episodes=6000)

# GATE -- do not proceed unless base separation is real
Zb_i = np.random.default_rng(0).choice(np.where(np.isin(y, BASE_IDX[:5]))[0], 5000, replace=False)
Zb, yb = embed(ext, X[Zb_i]), y[Zb_i]
base_acc = LogisticRegression(max_iter=2000).fit(Zb[:2500], yb[:2500]).score(Zb[2500:], yb[2500:])
print(f"logreg on base: {base_acc:.3f}  (chance 0.20; observed 0.67)")
assert base_acc > 0.35, "extractor did not learn -- everything downstream is noise"

sup, Xq, yq = novel_pools(band_mask(-8, 0))
Zq = embed(ext, Xq)

# ---------------------------------------------------------------- Cell 6
# THE HEADLINE RESULT.  Injectivity, not capacity, sets the ceiling.
g = np.random.default_rng(100)
pick = np.concatenate([g.choice(sup[c], 20, replace=False) for c in range(4)])
Zs, ys_ = embed(ext, X[pick]), np.repeat(np.arange(4), 20)

print(f"{'encoding':<16}{'test acc':>10}{'sd':>8}")
for nm, sc in [("pi (paper)", np.pi), ("pi/2 (V-E fix)", np.pi / 2),
               ("pi/4", np.pi / 4), ("pi/8", np.pi / 8)]:
    accs = [fit_head("quantum", Zs, ys_, Zq, yq, 4, seed=s, scale=sc) for s in range(5)]
    print(f"{nm:<16}{np.mean(accs):>10.3f}{np.std(accs):>8.3f}")
