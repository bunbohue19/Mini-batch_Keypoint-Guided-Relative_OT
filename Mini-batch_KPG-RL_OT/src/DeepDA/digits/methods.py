"""
Mini-batch Keypoint-Guided Optimal Transport for Deep Domain Adaptation (digits).

Extends the baseline Mini-batch-OT methods (mOT / BoMb-OT) with KPG-RL guidance:
  - Keypoint pairs are identified per mini-batch from source labels + target
    pseudo-labels (first sample per shared class).
  - A mask matrix enforces that keypoint pairs are matched exclusively.
  - A guiding matrix G (JSD of relation profiles to keypoints) steers the
    transport plan towards structurally consistent matchings.
  - The blended cost  alpha * C_norm + (1 - alpha) * G  follows the KPG-RL-KP
    formulation from Gu et al. (NeurIPS 2022).

When --use_kpg is not set, behaviour is identical to the baseline methods.py.
"""

import os

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils import model_eval, save_acc


# -----------------------------------------------------------------------
# KPG-RL helper functions
# -----------------------------------------------------------------------

def _softmax_rows(x):
    """Row-wise softmax (numerically stable)."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-20)


def _js_divergence_matrix(P, Q, eps=1e-10):
    """JSD between every row-pair of P (m, U) and Q (n, U) -> (m, n)."""
    P_e = P[:, np.newaxis, :]
    Q_e = Q[np.newaxis, :, :]
    M = 0.5 * (P_e + Q_e)
    kl1 = np.sum(P_e * (np.log(P_e + eps) - np.log(M + eps)), axis=-1)
    kl2 = np.sum(Q_e * (np.log(Q_e + eps) - np.log(M + eps)), axis=-1)
    return 0.5 * (kl1 + kl2)


def _sq_dist(A, B):
    """Squared Euclidean distance matrix between rows of A and B."""
    return (
        np.sum(A ** 2, axis=1, keepdims=True)
        + np.sum(B ** 2, axis=1, keepdims=True).T
        - 2.0 * A @ B.T
    )


def select_keypoints(ys_np, pred_xt_np, n_class):
    """Select keypoint pairs from source labels and target pseudo-labels.

    For each class present in both source and target, the first occurrence
    in each is taken as the representative keypoint.

    Returns (I_kp, J_kp) — lists of paired source/target indices.
    """
    pseudo_labels = pred_xt_np.argmax(axis=1)
    I_kp, J_kp = [], []
    for c in range(n_class):
        src_idx = np.where(ys_np == c)[0]
        tgt_idx = np.where(pseudo_labels == c)[0]
        if len(src_idx) > 0 and len(tgt_idx) > 0:
            I_kp.append(int(src_idx[0]))
            J_kp.append(int(tgt_idx[0]))
    return I_kp, J_kp


def build_mask(m, n, I_kp, J_kp):
    """KPG-RL binary mask (Proposition 1 of Gu et al.)."""
    Mask = np.ones((m, n), dtype=np.float64)
    for idx in I_kp:
        Mask[idx, :] = 0.0
    for jdx in J_kp:
        Mask[:, jdx] = 0.0
    for idx, jdx in zip(I_kp, J_kp):
        Mask[idx, jdx] = 1.0
    return Mask


def compute_guiding_matrix(feat_s, feat_t, I_kp, J_kp, tau_s=0.1, tau_t=0.1):
    """Compute G = JSD of relation profiles.  Matches original KPG-RL utils.py."""
    C_ss = _sq_dist(feat_s, feat_s)
    C_tt = _sq_dist(feat_t, feat_t)
    C_ss = C_ss / (C_ss.max() + 1e-10)
    C_tt = C_tt / (C_tt.max() + 1e-10)
    # -2* factor matches original: Rs = softmax_matrix(-2 * C1_kp / tau_s)
    R_s = _softmax_rows(-2.0 * C_ss[:, I_kp] / tau_s)
    R_t = _softmax_rows(-2.0 * C_tt[:, J_kp] / tau_t)
    return _js_divergence_matrix(R_s, R_t)


def sinkhorn_kpg_log(p, q, C, Mask, reg=0.01, niter=1000, thresh=1e-9):
    """Log-domain Sinkhorn with KPG-RL mask."""
    def log_kernel(u, v):
        lK = (-C + u[:, None] + v[None, :]) / reg
        lK[Mask == 0] = -1e20
        return lK

    def lse(A, axis):
        mx = np.max(A, axis=axis, keepdims=True)
        return np.log(np.exp(A - mx).sum(axis=axis, keepdims=True) + 1e-20) + mx

    u = np.zeros(len(p))
    v = np.zeros(len(q))
    for _ in range(niter):
        u_prev = u.copy()
        lk = log_kernel(u, v)
        u = reg * (np.log(p + 1e-20) - lse(lk, axis=1).squeeze()) + u
        lk = log_kernel(u, v)
        v = reg * (np.log(q + 1e-20) - lse(lk, axis=0).squeeze()) + v
        if np.linalg.norm(u - u_prev) < thresh:
            break
    return np.exp(log_kernel(u, v))


# -----------------------------------------------------------------------
# OT dispatch
# -----------------------------------------------------------------------

def _solve_ot(a, b, C_np, method, epsilon, tau, mass):
    """Standard OT solver (no KPG)."""
    if method == "jumbot":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_np, epsilon, tau)
    elif method == "jdot":
        if epsilon == 0:
            return ot.emd(a, b, C_np)
        else:
            return ot.sinkhorn(a, b, C_np, reg=epsilon)
    elif method == "jpmbot":
        if epsilon == 0:
            return ot.partial.partial_wasserstein(a, b, C_np, mass)
        else:
            _M = C_np / (C_np.max() + 1e-10)
            return ot.partial.entropic_partial_wasserstein(a, b, _M, m=mass, reg=epsilon)


def _solve_ot_kpg(a, b, C_np, method, epsilon, tau, mass,
                   feat_s, feat_t, ys_np, pred_xt_np, n_class,
                   tau_s, tau_t, alpha):
    """OT with KPG-RL-KP keypoint guidance.

    Blending: cost = alpha * C_norm + (1-alpha) * G
    G is NOT normalised (JSD is naturally bounded in [0, ln2]).
    Falls back to standard OT when no shared class exists in the batch.
    """
    I_kp, J_kp = select_keypoints(ys_np, pred_xt_np, n_class)
    if len(I_kp) == 0:
        return _solve_ot(a, b, C_np, method, epsilon, tau, mass)

    Mask = build_mask(len(a), len(b), I_kp, J_kp)
    G = compute_guiding_matrix(feat_s, feat_t, I_kp, J_kp, tau_s, tau_t)

    C_norm = C_np / (C_np.max() + 1e-10)
    M_kpg = alpha * C_norm + (1.0 - alpha) * G

    if method == "jdot" and epsilon > 0:
        return sinkhorn_kpg_log(a, b, M_kpg, Mask, reg=epsilon)
    elif method == "jdot" and epsilon == 0:
        M_masked = M_kpg.copy()
        M_masked[Mask == 0] = M_masked.max() * 1e3 + 1.0
        return ot.emd(a, b, M_masked)
    else:
        # unbalanced / partial: encode mask as large additive penalty
        M_masked = M_kpg.copy()
        M_masked[Mask == 0] = M_masked.max() * 1e3 + 1.0
        return _solve_ot(a, b, M_masked, method, epsilon, tau, mass)


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class DigitsDA:
    def __init__(
        self, model_g, model_f, n_class, logger, out_dir,
        eta1=0.1, eta2=0.1, epsilon=0.1, batch_epsilon=0.0,
        mass=0.5, tau=1.0, test_interval=10,
        use_kpg=False, alpha=0.5, tau_s=0.1, tau_t=0.1,
    ):
        self.model_g = model_g
        self.model_f = model_f
        self.n_class = n_class
        self.logger = logger
        self.out_dir = out_dir
        self.out_file = os.path.join(self.out_dir, "acc.csv")
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        self.eta1 = eta1
        self.eta2 = eta2
        self.epsilon = epsilon
        self.batch_epsilon = batch_epsilon
        self.mass = mass
        self.tau = tau
        self.test_interval = test_interval
        self.use_kpg = use_kpg
        self.alpha = alpha
        self.tau_s = tau_s
        self.tau_t = tau_t
        self.logger.info(
            "eta1={}, eta2={}, epsilon={}, use_kpg={}, alpha={}, tau_s={}, tau_t={}".format(
                eta1, eta2, epsilon, use_kpg, alpha, tau_s, tau_t
            )
        )

    # ----- inner OT helper (shared by all training paths) ---------------

    def _inner_ot(self, g_xs, g_xt, ys, pred_xt, total_cost, method):
        """Solve the inner OT and return the plan as a cuda tensor."""
        a = ot.unif(g_xs.size(0))
        b = ot.unif(g_xt.size(0))
        C_np = total_cost.detach().cpu().numpy()

        if self.use_kpg:
            pi = _solve_ot_kpg(
                a, b, C_np, method, self.epsilon, self.tau, self.mass,
                feat_s=g_xs.detach().cpu().numpy(),
                feat_t=g_xt.detach().cpu().numpy(),
                ys_np=ys.cpu().numpy(),
                pred_xt_np=pred_xt.detach().cpu().numpy(),
                n_class=self.n_class,
                tau_s=self.tau_s, tau_t=self.tau_t, alpha=self.alpha,
            )
        else:
            pi = _solve_ot(a, b, C_np, method, self.epsilon, self.tau, self.mass)

        return torch.from_numpy(pi).float().cuda()

    def _compute_cost(self, g_xs, g_xt, ys, pred_xt):
        """Compute ground cost: eta1 * embed + eta2 * label."""
        embed_cost = torch.cdist(g_xs, g_xt) ** 2
        ys_oh = F.one_hot(ys, num_classes=self.n_class).float()
        t_cost = -torch.mm(ys_oh, torch.log(pred_xt).T)
        return self.eta1 * embed_cost + self.eta2 * t_cost

    # ----- fit: standard mOT (averaging) --------------------------------

    def fit(self, source_loader, target_loader, test_loader,
            n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
            k=1, batch_size=25, method="jumbot"):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)

                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    total_loss = 0
                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()

                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    xt_mb = xt_mb_all[inds_xs[i]].cuda()
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                    pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost, method)

                    da_loss = 1.0 / k * torch.sum(pi * total_cost)
                    total_loss += da_loss
                    total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- fit_bomb: full outer-plan weighting --------------------------

    def fit_bomb(self, source_loader, target_loader, test_loader,
                 n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
                 k=1, batch_size=25, method="jumbot"):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)
                inds_xt = np.split(np.arange(xt_mb_all.shape[0]), k)

                # Phase 1: compute k*k cost matrix (no grad)
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                            pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost, method)
                            list_da_loss.append(torch.sum(pi * total_cost))

                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)

                # Phase 2: re-forward with grad
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    for j in range(k):
                        total_loss = 0
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        f_g_xs_mb = self.model_f(g_xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()

                        s_loss = 1.0 / (k ** 2) * criterion(f_g_xs_mb, ys)
                        total_loss += s_loss

                        if plan[i, j] == 0:
                            total_loss.backward()
                            continue

                        xt_mb = xt_mb_all[inds_xt[j]].cuda()
                        g_xt_mb = self.model_g(xt_mb)
                        f_g_xt_mb = self.model_f(g_xt_mb)
                        pred_xt = F.softmax(f_g_xt_mb, 1)

                        total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                        pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost, method)

                        da_loss = plan[i, j] * torch.sum(pi * total_cost)
                        total_loss += da_loss
                        total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- fit_bomb2: stable 1-to-1 matching variant --------------------

    def fit_bomb2(self, source_loader, target_loader, test_loader,
                  n_epochs, criterion=nn.CrossEntropyLoss(), lr=2e-4,
                  k=1, batch_size=25, method="jumbot"):
        criterion = nn.CrossEntropyLoss()
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        best_acc = 0

        for id_epoch in range(n_epochs):
            print(f"Epoch: {id_epoch}")
            self.model_g.train()
            self.model_f.train()
            target_loader_iter = iter(target_loader)

            for _, data in tqdm(enumerate(source_loader)):
                xs_mb_all, ys_all = data
                try:
                    xt_mb_all, _ = next(target_loader_iter)
                except StopIteration:
                    xt_mb_all = None
                if xt_mb_all is None or len(xt_mb_all) != batch_size:
                    target_loader_iter = iter(target_loader)
                    xt_mb_all, _ = next(target_loader_iter)

                inds_xs = np.split(np.arange(xs_mb_all.shape[0]), k)
                inds_xt = np.split(np.arange(xt_mb_all.shape[0]), k)

                # Phase 1: k*k cost + outer OT (no grad)
                list_da_loss = []
                with torch.no_grad():
                    for i in range(k):
                        xs_mb = xs_mb_all[inds_xs[i]].cuda()
                        g_xs_mb = self.model_g(xs_mb)
                        ys = ys_all[inds_xs[i]].cuda()
                        for j in range(k):
                            xt_mb = xt_mb_all[inds_xt[j]].cuda()
                            g_xt_mb = self.model_g(xt_mb)
                            f_g_xt_mb = self.model_f(g_xt_mb)
                            pred_xt = F.softmax(f_g_xt_mb, 1)

                            total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                            pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost, method)
                            list_da_loss.append(torch.sum(pi * total_cost))

                    big_C = torch.stack(list_da_loss).view(k, k)
                    if self.batch_epsilon == 0:
                        plan = ot.emd([], [], big_C.detach().cpu().numpy())
                    else:
                        plan = ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=self.batch_epsilon)
                    mapping = np.argmax(plan, axis=1)

                # Phase 2: re-forward only matched pairs (with grad)
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                for i in range(k):
                    j = mapping[i]
                    total_loss = 0

                    xs_mb = xs_mb_all[inds_xs[i]].cuda()
                    g_xs_mb = self.model_g(xs_mb)
                    f_g_xs_mb = self.model_f(g_xs_mb)
                    ys = ys_all[inds_xs[i]].cuda()

                    s_loss = 1.0 / k * criterion(f_g_xs_mb, ys)
                    total_loss += s_loss

                    xt_mb = xt_mb_all[inds_xt[j]].cuda()
                    g_xt_mb = self.model_g(xt_mb)
                    f_g_xt_mb = self.model_f(g_xt_mb)
                    pred_xt = F.softmax(f_g_xt_mb, 1)

                    total_cost = self._compute_cost(g_xs_mb, g_xt_mb, ys, pred_xt)
                    pi = self._inner_ot(g_xs_mb, g_xt_mb, ys, pred_xt, total_cost, method)

                    da_loss = plan[i, j] * torch.sum(pi * total_cost)
                    total_loss += da_loss
                    total_loss.backward()

                optimizer_g.step()
                optimizer_f.step()

            if id_epoch % self.test_interval == 0 or id_epoch == n_epochs - 1:
                source_acc = self.evaluate(source_loader)
                target_acc = self.evaluate(test_loader)
                self.logger.info(
                    "At epoch {} source and test accuracies are {} and {}".format(
                        id_epoch, source_acc, target_acc
                    )
                )
                save_acc(self.out_file, id_epoch, target_acc)
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save(
                        {"model_g": self.model_g.state_dict(),
                         "model_f": self.model_f.state_dict(),
                         "epoch": id_epoch, "accuracy": target_acc},
                        os.path.join(self.out_dir, "best_model.pth"),
                    )

        torch.save(
            {"model_g": self.model_g.state_dict(),
             "model_f": self.model_f.state_dict(),
             "epoch": n_epochs, "accuracy": target_acc},
            os.path.join(self.out_dir, "final_model.pth"),
        )

    # ----- source-only pre-training -------------------------------------

    def source_only(self, source_loader, criterion=nn.CrossEntropyLoss(), lr=2e-4):
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=lr)
        optimizer_f = torch.optim.Adam(self.model_f.parameters(), lr=lr)
        for _ in tqdm(range(10)):
            self.model_g.train()
            self.model_f.train()
            for _, data in enumerate(source_loader):
                xs_mb, ys = data
                xs_mb, ys = xs_mb.cuda(), ys.cuda()
                g_xs_mb = self.model_g(xs_mb)
                f_g_xs_mb = self.model_f(g_xs_mb)
                s_loss = criterion(f_g_xs_mb, ys)
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()
                s_loss.backward()
                optimizer_g.step()
                optimizer_f.step()
        source_acc = self.evaluate(source_loader)
        self.logger.info("Source accuracy is {}".format(source_acc))

    def evaluate(self, data_loader):
        return model_eval(data_loader, self.model_g, self.model_f)
