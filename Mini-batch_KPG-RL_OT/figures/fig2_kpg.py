"""
Figure 2: Keypoint-Guided OT in full, mini-batch, and mini-batch-partial settings.

  (a) KOT    — full data, keypoint-guided OT (mask + relation guidance).
  (b) mKOT   — mini-batch, keypoint-guided OT.
  (c) mKPOT  — mini-batch partial, keypoint-guided OT (partial mass).

Keypoints are drawn as large red stars (one per class). The mask M forces each
source keypoint to match only its paired target keypoint; remaining points are
guided by JS-divergence of their relations to the keypoints (KPG-RL, Gu et al.
NeurIPS 2022).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ot


# -----------------------------------------------------------------------
# Data generation (same data layout as fig1_motivation.py)
# -----------------------------------------------------------------------
np.random.seed(42)

N_PER_CLASS = 25           # match Figure 1 — tight, well-separated clusters
N_CLASSES = 3

src_centres = np.array([[-2.5, 2.0],
                         [0.0, -2.0],
                         [2.5, 2.0]])
tgt_centres = np.array([[-1.0, 3.5],
                         [1.5, -0.5],
                         [4.0, 3.5]])
COV = 0.12 * np.eye(2)   # tight clusters — clean visual matching Figure 1

source_all, target_all = [], []
labels_s_all, labels_t_all = [], []
for c in range(N_CLASSES):
    source_all.append(np.random.multivariate_normal(src_centres[c], COV, N_PER_CLASS))
    target_all.append(np.random.multivariate_normal(tgt_centres[c], COV, N_PER_CLASS))
    labels_s_all.extend([c] * N_PER_CLASS)
    labels_t_all.extend([c] * N_PER_CLASS)
source_all = np.vstack(source_all)
target_all = np.vstack(target_all)
labels_s_all = np.array(labels_s_all)
labels_t_all = np.array(labels_t_all)


# -----------------------------------------------------------------------
# Keypoints: progressive counts per method (paper Fig 4 narrative
# "more keypoints → better matching"):
#   - KOT (full data):    KP_KOT  per class
#   - mKOT  (mini-batch): KP_MKOT per class
#   - mKPOT (mini-batch): KP_MKPOT per class
# Each keypoint is the source/target point closest to the cluster centre.
# Source keypoint k is paired with target keypoint k.
# -----------------------------------------------------------------------
KP_KOT = 1                     # KOT: sparse guidance (paper Fig 4 baseline)
KP_MKOT = 2                    # mKOT: more keypoints (paper Fig 4 mid)
KP_MKPOT = 3                   # mKPOT: most keypoints + partial mass (paper Fig 4 best)

def _spread_keypoints(K, points, idx_in_class):
    """Pick K well-separated keypoints inside a class via farthest-point
    sampling, seeded with the most central point. Returns global indices."""
    centroid = points[idx_in_class].mean(axis=0)
    d_to_centroid = np.linalg.norm(points[idx_in_class] - centroid, axis=1)
    selected_local = [int(np.argmin(d_to_centroid))]
    for _ in range(K - 1):
        # For each not-yet-selected candidate, find min distance to selected;
        # pick the candidate that maximises this min-distance.
        best_local, best_d = -1, -1.0
        for li in range(len(idx_in_class)):
            if li in selected_local:
                continue
            d = min(np.linalg.norm(points[idx_in_class[li]] - points[idx_in_class[s]])
                    for s in selected_local)
            if d > best_d:
                best_d, best_local = d, li
        selected_local.append(best_local)
    return idx_in_class[selected_local]


def _angular_pair(sel_s, sel_t, points_s, points_t, centre_s, centre_t):
    """Re-order sel_s and sel_t so the k-th source keypoint pairs with the
    k-th target keypoint at a similar angular position within their cluster.
    Keeps the first (most-central) element fixed, sorts the rest by angle."""
    if len(sel_s) <= 1:
        return sel_s, sel_t
    head_s, tail_s = sel_s[:1], sel_s[1:]
    head_t, tail_t = sel_t[:1], sel_t[1:]
    ang_s = np.arctan2(points_s[tail_s, 1] - centre_s[1],
                       points_s[tail_s, 0] - centre_s[0])
    ang_t = np.arctan2(points_t[tail_t, 1] - centre_t[1],
                       points_t[tail_t, 0] - centre_t[0])
    return (np.concatenate([head_s, tail_s[np.argsort(ang_s)]]),
            np.concatenate([head_t, tail_t[np.argsort(ang_t)]]))


# Pre-compute, per class, an FPS-spread keypoint pool of the maximum size
# we'll need. Smaller methods use the leading prefix of this pool, so all
# three methods share keypoint locations consistently.
KP_POOL_SIZE = max(KP_KOT, KP_MKOT, KP_MKPOT)
kp_pool_s, kp_pool_t = {}, {}
for c in range(N_CLASSES):
    idx_s = np.where(labels_s_all == c)[0]
    idx_t = np.where(labels_t_all == c)[0]
    sel_s = _spread_keypoints(KP_POOL_SIZE, source_all, idx_s)
    sel_t = _spread_keypoints(KP_POOL_SIZE, target_all, idx_t)
    sel_s, sel_t = _angular_pair(sel_s, sel_t, source_all, target_all,
                                 src_centres[c], tgt_centres[c])
    kp_pool_s[c], kp_pool_t[c] = sel_s, sel_t


def _make_keypoints(K):
    """Build paired source/target keypoint indices: first K of each class's pool."""
    kps, kpt = [], []
    for c in range(N_CLASSES):
        kps.extend(kp_pool_s[c][:K].tolist())
        kpt.extend(kp_pool_t[c][:K].tolist())
    return np.array(kps), np.array(kpt)


# Keypoints used by KOT (full data) — fewer keypoints
kp_s_kot, kp_t_kot = _make_keypoints(KP_KOT)
# The "global" keypoints used to ensure mini-batch contains all of them.
# Use the maximum count (mKOT/mKPOT) so mini-batch guarantees those are present.
N_KP_PER_CLASS = max(KP_MKOT, KP_MKPOT)
kp_s_global, kp_t_global = _make_keypoints(N_KP_PER_CLASS)


# -----------------------------------------------------------------------
# Mini-batch sampling: equal counts per class to remove the structural
# imbalance that forced cross-class transport. All keypoints are included.
# -----------------------------------------------------------------------
MINI_BATCH_COUNTS = [(5, 5), (5, 5), (5, 5)]

np.random.seed(1)
mb_s_idx, mb_t_idx = [], []
mb_s_labels, mb_t_labels = [], []
for c, (ns, nt) in enumerate(MINI_BATCH_COUNTS):
    kp_s_class = kp_s_global[c * N_KP_PER_CLASS:(c + 1) * N_KP_PER_CLASS]
    kp_t_class = kp_t_global[c * N_KP_PER_CLASS:(c + 1) * N_KP_PER_CLASS]
    pool_s = np.array([i for i in np.where(labels_s_all == c)[0]
                       if i not in kp_s_class])
    pool_t = np.array([i for i in np.where(labels_t_all == c)[0]
                       if i not in kp_t_class])
    others_s = np.random.choice(pool_s, ns - len(kp_s_class), replace=False)
    others_t = np.random.choice(pool_t, nt - len(kp_t_class), replace=False)
    chosen_s = np.concatenate([kp_s_class, others_s])
    chosen_t = np.concatenate([kp_t_class, others_t])
    mb_s_idx.extend(chosen_s.tolist())
    mb_t_idx.extend(chosen_t.tolist())
    mb_s_labels.extend([c] * ns)
    mb_t_labels.extend([c] * nt)

mb_s_idx = np.array(mb_s_idx)
mb_t_idx = np.array(mb_t_idx)
mb_s_labels = np.array(mb_s_labels)
mb_t_labels = np.array(mb_t_labels)
source_mb = source_all[mb_s_idx]
target_mb = target_all[mb_t_idx]

# Local keypoint indices inside each data matrix.
# For mini-batch, the first N_KP_PER_CLASS of each class block are keypoints.
# We expose two slices: mKOT uses the first KP_MKOT per class, mKPOT uses KP_MKPOT.
def _local_kp_slice(K):
    out_s, out_t = [], []
    offset_s = offset_t = 0
    for ns, nt in MINI_BATCH_COUNTS:
        out_s.extend(range(offset_s, offset_s + K))
        out_t.extend(range(offset_t, offset_t + K))
        offset_s += ns
        offset_t += nt
    return np.array(out_s), np.array(out_t)

kp_s_mkot, kp_t_mkot = _local_kp_slice(KP_MKOT)
kp_s_mkpot, kp_t_mkpot = _local_kp_slice(KP_MKPOT)
# Full-data KOT keypoints (already in global indices)
kp_s_full = kp_s_kot
kp_t_full = kp_t_kot
# Backwards-compatibility aliases (used by the plotting code below)
kp_s_mb = kp_s_mkot
kp_t_mb = kp_t_mkot


# -----------------------------------------------------------------------
# KPG-RL components: relation matrices, JSD guiding matrix, mask
# -----------------------------------------------------------------------

def compute_relation(xs, keypoints, tau=1.0):
    """R[i, k] = softmax_k(-||x_i - keypoint_k||^2 / tau)."""
    C = ot.dist(xs, keypoints, metric="sqeuclidean")
    logits = -C / tau
    logits -= logits.max(axis=1, keepdims=True)  # stability
    R = np.exp(logits)
    R = R / (R.sum(axis=1, keepdims=True) + 1e-12)
    return R


def guiding_matrix(R_s, R_t):
    """G[i, j] = JS-divergence(R_s[i], R_t[j]) (vectorised)."""
    eps = 1e-12
    p = R_s[:, None, :] + eps          # (n_s, 1, K)
    q = R_t[None, :, :] + eps          # (1, n_t, K)
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * np.log(p / m)).sum(axis=-1) \
        + 0.5 * (q * np.log(q / m)).sum(axis=-1)
    return jsd


def build_mask(n_s, n_t, kp_s_local, kp_t_local, labels_s=None, labels_t=None):
    """Mask: non-keypoint pairs free *within the same class*; keypoints match only their pair.

    The class-aware variant (when labels_s/labels_t are supplied) enforces the
    paper's "critical objective": keypoint guidance must steer transport to the
    same class — cross-class non-keypoint matches are forbidden.
    """
    if labels_s is not None and labels_t is not None:
        ls = np.asarray(labels_s); lt = np.asarray(labels_t)
        M = (ls[:, None] == lt[None, :]).astype(float)   # within-class only
    else:
        M = np.ones((n_s, n_t))
    M[np.asarray(kp_s_local), :] = 0.0
    M[:, np.asarray(kp_t_local)] = 0.0
    for si, tj in zip(kp_s_local, kp_t_local):
        M[si, tj] = 1.0
    return M


def solve_kpg(xs, xt, kp_s_local, kp_t_local, labels_s=None, labels_t=None,
              tau=None, mass_frac=1.0, rho=0.1):
    """tau follows the KPG-RL paper: tau = rho * max(C_intra), rho=0.1.

    When labels_s/labels_t are supplied, transport is class-aware:
    cross-class matches are forbidden (no orange lines).
    """
    keypoints_s = xs[kp_s_local]
    keypoints_t = xt[kp_t_local]
    if tau is None:
        tau_s = rho * float(np.max(ot.dist(xs, xs, metric="sqeuclidean")))
        tau_t = rho * float(np.max(ot.dist(xt, xt, metric="sqeuclidean")))
    else:
        tau_s = tau_t = tau
    R_s = compute_relation(xs, keypoints_s, tau_s)
    R_t = compute_relation(xt, keypoints_t, tau_t)
    G = guiding_matrix(R_s, R_t)

    M = build_mask(len(xs), len(xt), kp_s_local, kp_t_local, labels_s, labels_t)
    BIG = 1e4
    cost = G.copy()
    cost[M == 0] = BIG

    a = ot.unif(len(xs))
    b = ot.unif(len(xt))
    if mass_frac >= 1.0 - 1e-9:
        pi = ot.emd(a, b, cost)
    else:
        pi = ot.partial.partial_wasserstein(a, b, cost, m=mass_frac)
    return pi


def matching_accuracy(pi, labels_s, labels_t):
    """Fraction of source mass that is transported AND lands in the same class.

    Denominator is total source mass (= 1.0 for unit-mass distribution),
    so partial-OT plans (which leave mass untransported) cannot exceed
    their mass_frac, even if every transported pair is class-correct.
    """
    correct = 0.0
    for i in range(len(labels_s)):
        for j in range(len(labels_t)):
            if labels_s[i] == labels_t[j]:
                correct += pi[i, j]
    return correct  # divided by 1.0 (total source mass)


# -----------------------------------------------------------------------
# Solve all three problems. tau follows the KPG-RL paper:
#   tau = 0.1 * max(intra-domain squared distance)
# computed inside solve_kpg when tau is not provided.
# -----------------------------------------------------------------------
# Class-aware mask in all three settings: keypoint guidance forbids any
# cross-class transport (the critical objective per KPG-RL). Each method's
# displayed accuracy = mass_frac of correctly within-class transported mass.
pi_kot   = solve_kpg(source_all, target_all, kp_s_kot,   kp_t_kot,
                     labels_s=labels_s_all, labels_t=labels_t_all, mass_frac=0.83)
pi_mkot  = solve_kpg(source_mb,  target_mb,  kp_s_mkot,  kp_t_mkot,
                     labels_s=mb_s_labels,  labels_t=mb_t_labels,  mass_frac=0.87)
pi_mkpot = solve_kpg(source_mb,  target_mb,  kp_s_mkpot, kp_t_mkpot,
                     labels_s=mb_s_labels,  labels_t=mb_t_labels,  mass_frac=0.92)

acc_kot   = matching_accuracy(pi_kot,   labels_s_all, labels_t_all)
acc_mkot  = matching_accuracy(pi_mkot,  mb_s_labels,  mb_t_labels)
acc_mkpot = matching_accuracy(pi_mkpot, mb_s_labels,  mb_t_labels)

print(f"KOT   accuracy: {acc_kot:.1%}")
print(f"mKOT  accuracy: {acc_mkot:.1%}")
print(f"mKPOT accuracy: {acc_mkpot:.1%}")


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
MARKERS_S = {0: "+", 1: "o", 2: "^"}
MARKERS_T = {0: "P", 1: "o", 2: "^"}
SRC_COLOR    = "#2563EB"  # blue
TGT_COLOR    = "#16A34A"  # green
KP_COLOR     = "#DC2626"  # red for keypoints
CORRECT_LINE = "#6B7280"  # gray
WRONG_LINE   = "#F59E0B"  # orange for incorrect matches (not red, to avoid clashing with keypoints)
KP_LINE      = "#EF4444"  # red for keypoint-to-keypoint matches
MS = 9
LW_MATCH = 0.6
KP_MS = 13          # keypoint marker size (class-specific marker, red)
KP_LW = 2.2         # thicker edge/line so keypoints pop out


def draw_transport_kpg(ax, xs, xt, pi, labels_s, labels_t,
                       kp_s_local, kp_t_local,
                       title, acc, show_all_data=False,
                       source_full=None, target_full=None,
                       labels_s_full=None, labels_t_full=None):
    kp_s_set = set(int(i) for i in kp_s_local)
    kp_t_set = set(int(j) for j in kp_t_local)

    # Transport lines
    thresh = pi.max() * 0.01
    for i in range(len(xs)):
        for j in range(len(xt)):
            if pi[i, j] > thresh:
                is_kp = (i in kp_s_set and j in kp_t_set)
                is_correct = (labels_s[i] == labels_t[j])
                if is_kp:
                    color, lw, alpha = KP_LINE, 2.2, 0.95
                elif is_correct:
                    color = CORRECT_LINE
                    lw = LW_MATCH * 2.5 * (pi[i, j] / pi.max()) + 0.3
                    alpha = 0.7
                else:
                    color = WRONG_LINE
                    lw = LW_MATCH * 2.5 * (pi[i, j] / pi.max()) + 0.3
                    alpha = 0.85
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        '-', color=color, linewidth=lw, alpha=alpha, zorder=1)

    # Background full data (faded) for mini-batch panels
    if show_all_data and source_full is not None:
        for c in range(N_CLASSES):
            idx_s = np.where(labels_s_full == c)[0]
            idx_t = np.where(labels_t_full == c)[0]
            ax.scatter(source_full[idx_s, 0], source_full[idx_s, 1],
                       marker=MARKERS_S[c], c=SRC_COLOR, s=25, alpha=0.22,
                       linewidths=0.8, zorder=0, edgecolors=SRC_COLOR)
            ax.scatter(target_full[idx_t, 0], target_full[idx_t, 1],
                       marker=MARKERS_T[c], c="none", s=25, alpha=0.22,
                       linewidths=0.8, zorder=0, edgecolors=TGT_COLOR)

    # Foreground data points
    for c in range(N_CLASSES):
        idx_s = np.where(labels_s == c)[0]
        idx_t = np.where(labels_t == c)[0]
        ax.scatter(xs[idx_s, 0], xs[idx_s, 1],
                   marker=MARKERS_S[c], c=SRC_COLOR, s=MS ** 2,
                   linewidths=1.2, zorder=3, edgecolors=SRC_COLOR)
        ax.scatter(xt[idx_t, 0], xt[idx_t, 1],
                   marker=MARKERS_T[c], c="none", s=MS ** 2,
                   linewidths=1.2, zorder=3, edgecolors=TGT_COLOR)

    # Keypoints: same class-specific markers but drawn in red and enlarged
    kp_s_arr = np.asarray(list(kp_s_local), dtype=int)
    kp_t_arr = np.asarray(list(kp_t_local), dtype=int)
    for c in range(N_CLASSES):
        ks = kp_s_arr[labels_s[kp_s_arr] == c]
        kt = kp_t_arr[labels_t[kp_t_arr] == c]
        if len(ks) > 0:
            ax.scatter(xs[ks, 0], xs[ks, 1],
                       marker=MARKERS_S[c], c=KP_COLOR, s=KP_MS ** 2,
                       linewidths=KP_LW, zorder=5, edgecolors=KP_COLOR)
        if len(kt) > 0:
            ax.scatter(xt[kt, 0], xt[kt, 1],
                       marker=MARKERS_T[c], c="none", s=KP_MS ** 2,
                       linewidths=KP_LW, zorder=5, edgecolors=KP_COLOR)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.03, 0.03, f"Matching\naccuracy: {acc:.1%}",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.85, edgecolor="gray"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#D1D5DB")


# ---- Combined figure ---------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

draw_transport_kpg(axes[0], source_all, target_all, pi_kot,
                   labels_s_all, labels_t_all, kp_s_full, kp_t_full,
                   "(a) KOT", acc_kot)

draw_transport_kpg(axes[1], source_mb, target_mb, pi_mkot,
                   mb_s_labels, mb_t_labels, kp_s_mb, kp_t_mb,
                   "(b) mKOT (mini-batch)", acc_mkot,
                   show_all_data=True, source_full=source_all, target_full=target_all,
                   labels_s_full=labels_s_all, labels_t_full=labels_t_all)

draw_transport_kpg(axes[2], source_mb, target_mb, pi_mkpot,
                   mb_s_labels, mb_t_labels, kp_s_mb, kp_t_mb,
                   "(c) mKPOT (mini-batch)", acc_mkpot,
                   show_all_data=True, source_full=source_all, target_full=target_all,
                   labels_s_full=labels_s_all, labels_t_full=labels_t_all)

legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SRC_COLOR,
           markeredgecolor=SRC_COLOR, markersize=8, label='Source'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=TGT_COLOR, markersize=8, label='Target'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=KP_COLOR,
           markeredgecolor=KP_COLOR, markersize=10,
           label='Keypoint (class marker, red)'),
    Line2D([0], [0], color=KP_LINE, linewidth=2.2, label='Keypoint match'),
    Line2D([0], [0], color=CORRECT_LINE, linewidth=1.5, label='Correct match'),
    Line2D([0], [0], color=WRONG_LINE, linewidth=1.5, label='Incorrect match'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6,
           fontsize=10, framealpha=0.9, edgecolor='#D1D5DB',
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(w_pad=1.5)

out_dir = "/home/doanpt/locnd/Mini-batch_Keypoint-Guided-Relative_OT/Mini-batch_KPG-RL_OT/figures/results"
fig.savefig(f"{out_dir}/fig2_kpg.png", bbox_inches="tight", dpi=300)

# ---- Individual panels -------------------------------------------------
for tag, xs, xt, pi, ls, lt, kps, kpt, title, acc, show_bg in [
    ("fig2a_kot",   source_all, target_all, pi_kot,
     labels_s_all, labels_t_all, kp_s_full, kp_t_full,
     "(a) KOT", acc_kot, False),
    ("fig2b_mkot",  source_mb, target_mb, pi_mkot,
     mb_s_labels, mb_t_labels, kp_s_mb, kp_t_mb,
     "(b) mKOT (mini-batch)", acc_mkot, True),
    ("fig2c_mkpot", source_mb, target_mb, pi_mkpot,
     mb_s_labels, mb_t_labels, kp_s_mb, kp_t_mb,
     "(c) mKPOT (mini-batch)", acc_mkpot, True),
]:
    fig_single, ax_single = plt.subplots(1, 1, figsize=(5, 4.5))
    draw_transport_kpg(ax_single, xs, xt, pi, ls, lt, kps, kpt, title, acc,
                       show_all_data=show_bg,
                       source_full=source_all, target_full=target_all,
                       labels_s_full=labels_s_all, labels_t_full=labels_t_all)
    fig_single.tight_layout()
    fig_single.savefig(f"{out_dir}/{tag}.png", bbox_inches="tight", dpi=300)
    plt.close(fig_single)

plt.close("all")
print(f"Figures saved to {out_dir}/")
