# =============================================================================
#  TWO EXPERIMENTS REQUESTED BY REVIEW
#  Run in the restored Colab session (Cells R1-R4 of the resume notebook).
#
#  E1  Does kappa scale with qubit count?      n = 6, 8, 10, 12
#  E2  Do the conclusions survive a different backbone?  ResNet extractor
#
#  Both are written so a NEGATIVE outcome is as informative as a positive one.
#  Neither is required for the current paper's claims, which are scoped to
#  n = 6 and one extractor family; they would REMOVE that scoping.
# =============================================================================
import numpy as np, torch, torch.nn as nn, pennylane as qml, pickle, time
from scipy import stats as sps
from sklearn.preprocessing import StandardScaler


# =============================================================================
#  E1 — SCALING OF THE TRANSMISSION COEFFICIENT WITH QUBIT COUNT
#
#  Proposition 1 says the endpoint identification z_i = +1 ~ z_i = -1 holds for
#  every n. It does NOT say the resulting accuracy cost is n-independent, so the
#  question is empirical.
#
#  Cost note: statevector simulation is O(2^n) in memory and the adjoint
#  gradient is O(2^n) per parameter, so n = 12 is roughly 64x the n = 6 cost per
#  circuit. Expect ~10 min (n=6), ~20 (n=8), ~50 (n=10), ~2.5 h (n=12) on CPU.
#  Run n = 6, 8, 10 first; add 12 only if the trend is unclear.
# =============================================================================
QUBIT_GRID = [6, 8, 10]        # append 12 if the trend is ambiguous
SEEDS_E1   = [0, 1, 2]
K_EVAL     = 20


def build_quantum_head(n_q, n_classes, n_layers=4):
    """VQC head at arbitrary width. Identical structure to the paper's n=6 head."""
    try:
        dev = qml.device("lightning.qubit", wires=n_q); diff = "adjoint"
    except Exception:
        dev = qml.device("default.qubit", wires=n_q); diff = "backprop"

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def qnode(inputs, weights):
        for l in range(n_layers):
            qml.AngleEmbedding(inputs, wires=range(n_q))
            qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_q))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    shapes = {"weights": (n_layers, n_q, 3)}

    class QHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qml.qnn.TorchLayer(qnode, shapes)
            self.out = nn.Linear(n_q, n_classes)
        def forward(self, x):
            return self.out(self.qlayer(x * np.pi))
    return QHead


def run_E1(train_extractor_at_width, extract_fn, train_head_fn,
           Xtr_h, ytr_h, Fq_builder, yq, n_classes):
    """
    train_extractor_at_width(d, seed) -> frozen extractor with d-dim bottleneck
    extract_fn(ext, X)                -> features
    train_head_fn(make_head, Fk, yk, Fq, yq) -> accuracy
    Fq_builder(F_test)                -> fixed query subset
    """
    res = {n: {"q": [], "c": []} for n in QUBIT_GRID}
    t0 = time.time()
    for n_q in QUBIT_GRID:
        QHead = build_quantum_head(n_q, n_classes)
        for seed in SEEDS_E1:
            ext = train_extractor_at_width(n_q, seed)          # bottleneck = n_q
            Ftr = extract_fn(ext, Xtr_h)
            Fq  = Fq_builder(extract_fn(ext, Xte_h))
            rng = np.random.default_rng(seed)
            idx = np.concatenate([rng.choice(np.where(ytr_h == c)[0],
                                             size=K_EVAL, replace=False)
                                  for c in range(n_classes)])
            Fk, yk = Ftr[idx], ytr_h[idx]
            torch.manual_seed(seed)
            a_q = train_head_fn(lambda: QHead(), Fk, yk, Fq, yq)
            torch.manual_seed(seed)
            a_c = train_head_fn(lambda: ClassicalHead(n_q, n_classes), Fk, yk, Fq, yq)
            res[n_q]["q"].append(a_q); res[n_q]["c"].append(a_c)
            print(f"  n={n_q:2d} seed {seed} | q {a_q:.3f} | c {a_c:.3f} "
                  f"| kappa {a_q/a_c:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== E1: does kappa depend on qubit count? ===")
    print("  n |   quantum   |  classical  |  kappa")
    ks = []
    for n_q in QUBIT_GRID:
        q = np.array(res[n_q]["q"]); c = np.array(res[n_q]["c"])
        k = float(np.mean(q / c)); ks.append(k)
        print(f" {n_q:2d} | {q.mean():.3f}±{q.std():.3f} | {c.mean():.3f}±{c.std():.3f} "
              f"| {k:.3f}±{np.std(q/c):.3f}")
    slope, _, r, p, _ = sps.linregress(QUBIT_GRID, ks)
    print(f"\n  kappa vs n: slope {slope:+.4f} per qubit, r={r:+.3f}, p={p:.3f}")
    print(f"  paper value at n=6: 0.641")
    if p < 0.05 and slope > 0:
        print("  -> kappa INCREASES with n: the n=6 result is a small-scale effect.")
        print("     Extrapolated n for kappa=0.9:",
              round((0.9 - ks[0]) / slope + QUBIT_GRID[0], 1))
    elif p < 0.05 and slope < 0:
        print("  -> kappa DECREASES with n: the interface cost worsens with width.")
    else:
        print("  -> kappa FLAT within noise: consistent with a structural constant,")
        print("     which is what Proposition 1 would predict if the identification")
        print("     cost is width-independent. This REMOVES the n=6 scoping caveat.")
    pickle.dump(res, open("results_E1_qubit_scaling.pkl", "wb"))
    return res


# =============================================================================
#  E2 — DOES THE CONCLUSION SURVIVE A DIFFERENT BACKBONE?
#
#  The paper evaluates one convolutional extractor. If kappa is a property of the
#  INTERFACE it should be roughly invariant to the encoder that feeds it; if it
#  is a property of that particular CNN, it should move.
# =============================================================================
class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1); self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1); self.b2 = nn.BatchNorm1d(ch)
    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(h)))


class ResNetExtractor(nn.Module):
    """ResNet-style encoder, same 6-dim tanh bottleneck as the paper's CNN."""
    def __init__(self, d, ch=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(2, ch, 7, padding=3),
                                  nn.BatchNorm1d(ch), nn.ReLU())
        blocks = []
        for _ in range(3):
            blocks += [ResBlock1D(ch), nn.MaxPool1d(2)]
        self.body = nn.Sequential(*blocks)                       # 128 -> 16
        self.bottleneck = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                        nn.Linear(ch, d), nn.Tanh())
    def forward(self, x):
        return self.bottleneck(self.body(self.stem(x)))


class TransformerExtractor(nn.Module):
    """Transformer encoder variant, same bottleneck. Use if a second backbone is wanted."""
    def __init__(self, d, dm=64, nhead=4, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(2, dm)
        self.pos = nn.Parameter(torch.randn(1, 128, dm) * 0.02)
        layer = nn.TransformerEncoderLayer(dm, nhead, dm * 2, 0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.bottleneck = nn.Sequential(nn.Linear(dm, d), nn.Tanh())
    def forward(self, x):
        h = self.proj(x.transpose(1, 2)) + self.pos
        return self.bottleneck(self.enc(h).mean(1))


def run_E2(pretrain_fn, extract_fn, train_head_fn, backbones,
           Xtr_h, ytr_h, Xte_h, yte_h, n_classes, n_q=6):
    """
    backbones: dict name -> constructor(d) returning an nn.Module extractor.
    pretrain_fn(model, base_classes, seed) -> frozen, prototypically pretrained.
    Reports kappa per backbone. Invariance supports an interface explanation.
    """
    out = {}
    for name, ctor in backbones.items():
        qs, cs = [], []
        for seed in [0, 1, 2]:
            ext = pretrain_fn(ctor(n_q), base_c, seed)
            Ftr, Fte = extract_fn(ext, Xtr_h), extract_fn(ext, Xte_h)
            rng = np.random.default_rng(seed)
            qi = rng.choice(len(Fte), size=1000, replace=False)
            Fq, yq = Fte[qi], yte_h[qi]
            idx = np.concatenate([rng.choice(np.where(ytr_h == c)[0], size=20,
                                             replace=False)
                                  for c in range(n_classes)])
            Fk, yk = Ftr[idx], ytr_h[idx]
            torch.manual_seed(seed)
            a_q = train_head_fn(lambda: QuantumHeadRU(n_q, n_classes, 4), Fk, yk, Fq, yq)
            torch.manual_seed(seed)
            a_c = train_head_fn(lambda: ClassicalHead(n_q, n_classes), Fk, yk, Fq, yq)
            qs.append(a_q); cs.append(a_c)
            print(f"  {name:12s} seed {seed} | q {a_q:.3f} | c {a_c:.3f} "
                  f"| kappa {a_q/a_c:.3f}", flush=True)
        out[name] = {"q": qs, "c": cs, "kappa": float(np.mean(np.array(qs)/np.array(cs)))}

    print("\n=== E2: is kappa a property of the interface or of the CNN? ===")
    print("  backbone     |   quantum   |  classical  | kappa")
    for name, r in out.items():
        q, c = np.array(r["q"]), np.array(r["c"])
        print(f"  {name:12s} | {q.mean():.3f}±{q.std():.3f} | "
              f"{c.mean():.3f}±{c.std():.3f} | {r['kappa']:.3f}")
    kk = [r["kappa"] for r in out.values()]
    spread = max(kk) - min(kk)
    print(f"\n  spread across backbones: {spread:.3f}   (paper CNN value 0.641)")
    if spread < 0.08:
        print("  -> kappa is backbone-INVARIANT: supports the interface explanation")
        print("     and removes the single-extractor scoping caveat.")
    else:
        print("  -> kappa VARIES with backbone: it is partly a property of the")
        print("     encoder, and the paper's value should stay scoped to its CNN.")
    pickle.dump(out, open("results_E2_backbones.pkl", "wb"))
    return out


# =============================================================================
#  HOW TO REPORT
#
#  E1 flat + E2 invariant   -> delete both scoping caveats in Limitations,
#                              promote kappa to a property of angle encoding.
#  E1 rising                -> keep the n=6 scope; report the extrapolated n at
#                              which kappa reaches 0.9, which is a useful number.
#  E2 varying               -> keep "one extractor family" and report the range.
#
#  Any of these is publishable. The failure mode to avoid is running only the
#  configuration that gives the preferred answer.
# =============================================================================


# =============================================================================
#  E3 — SECOND DATASET (the reviewer's largest remaining objection)
#
#  RadioML 2018.01A: 24 modulation classes, 1024 I/Q samples, SNR -20..+30 dB,
#  ~2.5M examples, 21 GB HDF5.  It differs from 2016.10a on every axis that
#  matters here: alphabet size, sequence length, SNR range, and impairments.
#
#  If kappa is a property of the ENCODING it should survive all of that.
#  If it moves, it is partly a property of 2016.10a and must stay scoped.
#
#  Practical notes:
#    - 21 GB will not fit in a free Colab session. Stream a stratified subset
#      with h5py rather than loading the file; ~120k examples is ample.
#    - Sequences are 1024 long, so the extractor needs one more pooling stage.
#    - Chance is 1/24 = 0.042, so accuracies are NOT comparable in absolute
#      terms to the 11-class results. Only kappa = q/c is comparable.
# =============================================================================
def load_2018_subset(h5_path, per_class_per_snr=60, snr_min=0, seed=42):
    """Stream a stratified subset of RadioML 2018.01A without loading 21 GB."""
    import h5py
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        Y = f["Y"][:]                       # (N, 24) one-hot
        Z = f["Z"][:].ravel()               # (N,) SNR
        lab = Y.argmax(1)
        keep = []
        for c in range(24):
            for snr in np.unique(Z):
                if snr < snr_min: continue
                idx = np.where((lab == c) & (Z == snr))[0]
                if len(idx):
                    keep.append(rng.choice(idx, size=min(per_class_per_snr, len(idx)),
                                           replace=False))
        keep = np.sort(np.concatenate(keep))
        X = f["X"][keep]                    # (M, 1024, 2)
        y = lab[keep]; snr = Z[keep]
    X = X.transpose(0, 2, 1).astype(np.float32)          # -> (M, 2, 1024)
    X = X / (np.sqrt((X**2).sum(axis=(1, 2), keepdims=True)) + 1e-8)
    print(f"2018.01A subset: {X.shape}, {len(np.unique(y))} classes, "
          f"SNR {snr.min()}..{snr.max()} dB")
    return X, y, snr


class Extractor1024(nn.Module):
    """Same design as the paper's extractor, one extra pool stage for L=1024."""
    def __init__(self, d, ch=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, ch, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.3),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.3))
        self.bottleneck = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                        nn.Linear(ch, d), nn.Tanh())
    def forward(self, x):
        return self.bottleneck(self.features(x))


def run_E3(pretrain_fn, extract_fn, train_head_fn,
           X18, y18, base_c18, novel_c18, n_q=6, seeds=(0, 1, 2)):
    """
    Same protocol as the paper: prototypical pretraining on base classes,
    few-shot the held-out novel classes, quantum vs matched classical head.
    Report kappa and compare with the 2016.10a value of 0.641.
    """
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X18, y18, test_size=0.3,
                                          random_state=42, stratify=y18)
    remap = {c: i for i, c in enumerate(novel_c18)}
    ntr = np.where(np.isin(ytr, novel_c18))[0]; nte = np.where(np.isin(yte, novel_c18))[0]
    Xtr_n = Xtr[ntr]; ytr_n = np.array([remap[c] for c in ytr[ntr]])
    Xte_n = Xte[nte]; yte_n = np.array([remap[c] for c in yte[nte]])
    C = len(novel_c18)

    qs, cs = [], []
    for seed in seeds:
        ext = pretrain_fn(Extractor1024(n_q), base_c18, seed)
        Ftr, Fte = extract_fn(ext, Xtr_n), extract_fn(ext, Xte_n)
        rng = np.random.default_rng(seed)
        qi = rng.choice(len(Fte), size=min(1000, len(Fte)), replace=False)
        Fq, yq = Fte[qi], yte_n[qi]
        idx = np.concatenate([rng.choice(np.where(ytr_n == c)[0], size=20, replace=False)
                              for c in range(C)])
        Fk, yk = Ftr[idx], ytr_n[idx]
        torch.manual_seed(seed)
        a_q = train_head_fn(lambda: QuantumHeadRU(n_q, C, 4), Fk, yk, Fq, yq)
        torch.manual_seed(seed)
        a_c = train_head_fn(lambda: ClassicalHead(n_q, C), Fk, yk, Fq, yq)
        qs.append(a_q); cs.append(a_c)
        print(f"  seed {seed} | q {a_q:.3f} | c {a_c:.3f} | kappa {a_q/a_c:.3f}", flush=True)

    q, c = np.array(qs), np.array(cs)
    kappa = float(np.mean(q / c))
    print(f"\n=== E3: does kappa transfer to RadioML 2018.01A? ===")
    print(f"  quantum   {q.mean():.3f} +- {q.std():.3f}")
    print(f"  classical {c.mean():.3f} +- {c.std():.3f}   (chance = {1/C:.3f})")
    print(f"  kappa     {kappa:.3f} +- {np.std(q/c):.3f}")
    print(f"  2016.10a  0.641 +- 0.032")
    d = abs(kappa - 0.641)
    if d < 0.06:
        print("  -> kappa TRANSFERS across datasets. This is the single strongest")
        print("     available evidence that it is a property of the encoding rather")
        print("     than of one corpus, and removes the largest scoping caveat.")
    else:
        print(f"  -> kappa DIFFERS by {d:.3f}. It is partly dataset-dependent and the")
        print("     paper's value must stay scoped to 2016.10a. Report both.")
    pickle.dump({"q": qs, "c": cs, "kappa": kappa},
                open("results_E3_dataset2.pkl", "wb"))
    return kappa


# =============================================================================
#  PRIORITY, given that all three cost real compute:
#
#    E3 (second dataset)   highest value, addresses the largest objection
#    E1 (qubit scaling)    next; also tests Proposition 1's n-independence
#    E2 (second backbone)  cheapest, but the weakest of the three objections
#
#  A single figure with kappa on the y-axis and {2016.10a CNN, 2018.01A CNN,
#  2016.10a ResNet, n=8, n=10} on the x-axis would answer all three at once and
#  would be the natural centrepiece of the journal version.
# =============================================================================
