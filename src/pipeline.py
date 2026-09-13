# ============================================================================
#  RESUME SCRIPT — rebuild the whole project state in one shot
#  Paste this entire file into a fresh Colab cell and run it top to bottom.
#  (Colab wipes memory when the runtime resets; this restores everything.)
#
#  Sections 1–6 rebuild state (a few minutes; the extractor pretraining is the
#  main cost). Section 7 is the SLOW sweep. Section 8 plots. Run 1–6, then 7, then 8.
# ============================================================================

# ---- Section 1: install + imports + seeds ----------------------------------
import subprocess, sys
for pkg in ["pennylane", "pennylane-lightning"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

import os, urllib.request, tarfile, pickle, time
import numpy as np
import torch, torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
Ks    = [1, 5, 10, 20]     # sweep grid (expand later if quantum is competitive)
seeds = [0, 1, 2]
print("device:", device)

# ---- Section 2: get the dataset (skip if already on disk) -------------------
PKL = "RML2016.10a_dict.pkl"
if not os.path.exists(PKL):
    print("downloading RadioML 2016.10a from Zenodo ...")
    urllib.request.urlretrieve(
        "https://zenodo.org/records/18397070/files/RML2016.10a.tar.bz2?download=1",
        "RML2016.10a.tar.bz2")
    with tarfile.open("RML2016.10a.tar.bz2", "r:bz2") as t:
        t.extractall()
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith("RML2016.10a_dict.pkl"):
                PKL = os.path.join(root, f)
print("dataset file:", PKL)

# ---- Section 3: load -> arrays X, y, S -------------------------------------
with open(PKL, "rb") as f:
    raw = pickle.load(f, encoding="latin1")
mods = sorted({k[0] for k in raw.keys()})
mod_to_idx = {m: i for i, m in enumerate(mods)}
X, y, S = [], [], []
for (mod, snr), arr in raw.items():
    X.append(arr); y += [mod_to_idx[mod]] * arr.shape[0]; S += [snr] * arr.shape[0]
X = np.vstack(X).astype(np.float32); y = np.array(y); S = np.array(S)
# per-sample L2 normalize
X = X / (np.sqrt((X**2).sum(axis=(1, 2), keepdims=True)) + 1e-8)
print("X:", X.shape, "| classes:", len(mods))

# ---- Section 4: split -------------------------------------------------------
Xtr, Xtmp, ytr, ytmp, Str, Stmp = train_test_split(
    X, y, S, test_size=0.4, random_state=SEED, stratify=y)
Xva, Xte, yva, yte, Sva, Ste = train_test_split(
    Xtmp, ytmp, Stmp, test_size=0.5, random_state=SEED, stratify=ytmp)

# ---- Section 5: models ------------------------------------------------------
n_qubits = 6
n_layers = 4

class FeatureExtractor(nn.Module):
    def __init__(self, d, L=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3))
        self.bottleneck = nn.Sequential(nn.Flatten(), nn.Linear(32*32, d), nn.Tanh())
    def forward(self, x): return self.bottleneck(self.features(x))

class ClassicalHead(nn.Module):
    def __init__(self, d, n_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
    def forward(self, x): return self.net(x)

class LinearHead(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__(); self.net = nn.Linear(d, n_classes)
    def forward(self, x): return self.net(x)

# quantum head with DATA RE-UPLOADING (lightning+adjoint; fallback default.qubit+backprop)
try:
    qdev = qml.device("lightning.qubit", wires=n_qubits); QDIFF = "adjoint"
except Exception:
    qdev = qml.device("default.qubit", wires=n_qubits); QDIFF = "backprop"
print("quantum device:", qdev.name, "| diff:", QDIFF)

@qml.qnode(qdev, interface="torch", diff_method=QDIFF)
def qnode_ru(inputs, weights):
    for l in range(n_layers):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights[l:l+1], wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
ru_shapes = {"weights": (n_layers, n_qubits, 3)}

class QuantumHeadRU(nn.Module):
    def __init__(self, n_qubits, n_classes, n_layers):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode_ru, ru_shapes)
        self.out = nn.Linear(n_qubits, n_classes)
    def forward(self, x): return self.out(self.qlayer(x * np.pi))

# ---- Section 6: pretrain extractor, freeze, extract frozen features ---------
tr_pool = np.where(Str >= 0)[0]; Xtp, ytp = Xtr[tr_pool], ytr[tr_pool]
te_pool = np.where(Ste >= 0)[0]; Xep, yep = Xte[te_pool], yte[te_pool]

extractor = FeatureExtractor(n_qubits).to(device)
probe     = nn.Linear(n_qubits, len(mods)).to(device)
opt  = torch.optim.Adam(list(extractor.parameters()) + list(probe.parameters()), lr=1e-3)
crit = nn.CrossEntropyLoss()
ds = torch.utils.data.TensorDataset(torch.tensor(Xtp), torch.tensor(ytp, dtype=torch.long))
ld = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
print("pretraining extractor ...")
for ep in range(20):
    extractor.train()
    for xb, yb in ld:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); crit(probe(extractor(xb)), yb).backward(); opt.step()

extractor.eval()
for p in extractor.parameters(): p.requires_grad_(False)
def extract(Xa, bs=1024):
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xa), bs):
            outs.append(extractor(torch.tensor(Xa[i:i+bs]).to(device)).cpu().numpy())
    return np.concatenate(outs)
Ftr, Fte = extract(Xtp), extract(Xep)
rng0 = np.random.default_rng(123)
ei = rng0.choice(len(Fte), size=1000, replace=False)   # 1000 keeps quantum eval fast
Fte_eval, yte_eval = Fte[ei], yep[ei]
print("features ready. Ftr:", Ftr.shape, "| Fte_eval:", Fte_eval.shape)

# head trainer (uses Fte_eval / yte_eval globals; all on CPU)
def train_head(make_head, Fk, yk, epochs=30, lr=1e-2):
    head = make_head()
    Xk = torch.tensor(Fk, dtype=torch.float32); Yk = torch.tensor(yk, dtype=torch.long)
    opt = torch.optim.Adam(head.parameters(), lr=lr); crit = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        opt.zero_grad(); crit(head(Xk), Yk).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(torch.tensor(Fte_eval, dtype=torch.float32)).argmax(1).numpy()
    return (pred == yte_eval).mean()

print("\n=== STATE REBUILT. Run Section 7 (the sweep) next. ===")

# ============================================================================
# ---- Section 7: THE SWEEP (slow — the quantum head dominates the time) ------
# ============================================================================
res2 = {"classical_mlp": {k: [] for k in Ks},
        "linear":        {k: [] for k in Ks},
        "quantum_ru":    {k: [] for k in Ks}}
print("starting sweep ...", flush=True)
for seed in seeds:
    rng = np.random.default_rng(seed)
    for K in Ks:
        idx = np.concatenate([rng.choice(np.where(ytp == c)[0], size=K, replace=False)
                              for c in range(len(mods))])
        Fk, yk = Ftr[idx], ytp[idx]
        torch.manual_seed(seed); res2["classical_mlp"][K].append(
            train_head(lambda: ClassicalHead(n_qubits, len(mods)), Fk, yk))
        torch.manual_seed(seed); res2["linear"][K].append(
            train_head(lambda: LinearHead(n_qubits, len(mods)), Fk, yk))
        tq = time.time()
        torch.manual_seed(seed); res2["quantum_ru"][K].append(
            train_head(lambda: QuantumHeadRU(n_qubits, len(mods), n_layers), Fk, yk))
        print(f"  seed {seed} K={K:2d} | quantum {time.time()-tq:5.1f}s", flush=True)
    print(f"seed {seed} DONE", flush=True)

# ============================================================================
# ---- Section 8: plot + K-table ---------------------------------------------
# ============================================================================
def st(d): return (np.array([np.mean(d[k]) for k in Ks]), np.array([np.std(d[k]) for k in Ks]))
for name, marker in [("classical_mlp", "o"), ("linear", "^"), ("quantum_ru", "s")]:
    m, s = st(res2[name]); plt.errorbar(Ks, m, yerr=s, marker=marker, capsize=4, label=name)
plt.axhline(1/len(mods), ls='--', c='gray', label='random')
plt.xlabel("examples per class (K)"); plt.ylabel("test accuracy")
plt.title("Experiment 1 v2 — quantum-RU vs classical (MLP & linear)")
plt.legend(); plt.grid(True); plt.show()
for k in Ks:
    print(f"K={k:2d} | mlp {np.mean(res2['classical_mlp'][k]):.3f} "
          f"| linear {np.mean(res2['linear'][k]):.3f} "
          f"| quantum {np.mean(res2['quantum_ru'][k]):.3f}")
