import argparse
import os
import random
import data_list
import lr_schedule
import network
import numpy as np
import ot
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import BalancedBatchSampler


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_classification(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader["test"])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = (
        torch.sum(torch.squeeze(predict).float() == all_label).item()
        / float(all_label.size()[0])
    )
    return accuracy


# -----------------------------------------------------------------------
# KPG-RL helper functions  (identical to DeepDA/office/train.py)
# -----------------------------------------------------------------------

def build_kpg_mask(m, n, I_kp, J_kp):
    """KPG-RL binary mask: keypoint rows/cols zeroed out, diagonal kp restored."""
    Mask = np.ones((m, n), dtype=np.float64)
    for idx in I_kp:
        Mask[idx, :] = 0.0
    for jdx in J_kp:
        Mask[:, jdx] = 0.0
    for idx, jdx in zip(I_kp, J_kp):
        Mask[idx, jdx] = 1.0
    return Mask


def _softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-20)


def _js_divergence_matrix(P, Q, eps=1e-10):
    P_e = P[:, np.newaxis, :]
    Q_e = Q[np.newaxis, :, :]
    M   = 0.5 * (P_e + Q_e)
    kl1 = np.sum(P_e * (np.log(P_e + eps) - np.log(M + eps)), axis=-1)
    kl2 = np.sum(Q_e * (np.log(Q_e + eps) - np.log(M + eps)), axis=-1)
    return 0.5 * (kl1 + kl2)


def compute_guiding_matrix(feat_s, feat_t, I_kp, J_kp, tau_s=0.1, tau_t=0.1):
    """KPG-RL guiding matrix G via JSD of relation profiles in feature space."""
    def sq_dist(A, B):
        return (
            np.sum(A ** 2, axis=1, keepdims=True)
            + np.sum(B ** 2, axis=1, keepdims=True).T
            - 2.0 * A @ B.T
        )

    C_ss = sq_dist(feat_s, feat_s)
    C_tt = sq_dist(feat_t, feat_t)
    C_ss = C_ss / (C_ss.max() + 1e-10)
    C_tt = C_tt / (C_tt.max() + 1e-10)

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
        max_A = np.max(A, axis=axis, keepdims=True)
        return np.log(np.exp(A - max_A).sum(axis=axis, keepdims=True) + 1e-20) + max_A

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


def select_keypoints_from_batch(
    ys_np, pred_xt_np, class_num, conf_thresh=0.0, n_shared_classes=None
):
    """Select paired keypoints: one per class present in both source and target.

    PDA-specific note
    -----------------
    In closed-set DA the source and target share the same label space, so the
    overlap between source labels and target pseudo-labels is automatically
    correct.  In Partial DA this is NOT true: the target only contains the
    first `n_shared_classes` classes (labels 0..n_shared_classes-1), but the
    classifier predicts over all `class_num` classes and may mistakenly assign
    a *source-private* class to some target sample early in training.  Pairing
    a source-private keypoint with such a target would force a wrong match
    via the KPG mask.

    When `n_shared_classes` is set (e.g., 25 for Office-Home PDA), candidate
    keypoint classes are restricted to 0..n_shared_classes-1 — preventing the
    wrong pairings above.  When None, all `class_num` classes are considered
    (closed-set behavior).
    """
    if n_shared_classes is None:
        n_shared_classes = class_num
    pseudo_conf = pred_xt_np.max(axis=1)
    pseudo_labels = pred_xt_np.argmax(axis=1)
    I_kp, J_kp = [], []
    for c in range(n_shared_classes):
        src_idx = np.where(ys_np == c)[0]
        tgt_idx = np.where(pseudo_labels == c)[0]
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            continue
        best_t = int(tgt_idx[np.argmax(pseudo_conf[tgt_idx])])
        if pseudo_conf[best_t] < conf_thresh:
            continue
        I_kp.append(int(src_idx[0]))
        J_kp.append(best_t)
    return I_kp, J_kp


# -----------------------------------------------------------------------
# OT solvers
# -----------------------------------------------------------------------

def solve_ot(a, b, M_norm, ot_type, epsilon, tau, adap_mass):
    """Standard OT dispatcher (ot / uot / pot)."""
    if ot_type == "ot":
        if epsilon == 0:
            return ot.emd(a, b, M_norm)
        else:
            return ot.sinkhorn(a, b, M_norm, reg=epsilon)
    elif ot_type == "uot":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_norm, epsilon, tau)
    elif ot_type == "pot":
        if epsilon == 0:
            return ot.partial.partial_wasserstein(a, b, M_norm, adap_mass)
        else:
            return ot.partial.entropic_partial_wasserstein(a, b, M_norm, m=adap_mass, reg=epsilon)


def solve_ot_kpg(
    a, b, M_cpu, ot_type, epsilon, tau, adap_mass,
    feat_s, feat_t, ys_np, pred_xt_np, class_num,
    tau_s, tau_t, alpha, n_shared_classes=None,
):
    """Mini-batch OT with KPG-RL-KP guidance.

    Blending follows the original KPG-RL-KP paper:
        C_norm = C / max(C)
        cost   = alpha * C_norm + (1 - alpha) * G

    Only C is normalised; G (JSD, bounded in [0, ln2]) is left as-is.
    Falls back to standard OT when no shared class is found in the batch.

    `n_shared_classes` restricts keypoint candidate classes for PDA — see
    `select_keypoints_from_batch` for the rationale.
    """
    I_kp, J_kp = select_keypoints_from_batch(
        ys_np, pred_xt_np, class_num, n_shared_classes=n_shared_classes
    )

    if len(I_kp) == 0:
        M_norm = M_cpu / (M_cpu.max() + 1e-8)
        return solve_ot(a, b, M_norm, ot_type, epsilon, tau, adap_mass)

    Mask  = build_kpg_mask(len(a), len(b), I_kp, J_kp)
    G     = compute_guiding_matrix(feat_s, feat_t, I_kp, J_kp, tau_s, tau_t)
    C_norm = M_cpu / (M_cpu.max() + 1e-8)
    M_kpg  = alpha * C_norm + (1.0 - alpha) * G

    if ot_type == "ot" and epsilon > 0:
        # Exact masked log-Sinkhorn for balanced + entropic case
        return sinkhorn_kpg_log(a, b, M_kpg, Mask, reg=epsilon)
    else:
        # Mask-as-penalty: exp(-pen/eps) stays safely positive in float64.
        # For epsilon>0: pen = max(2, 50*eps) → exp(-50) ≈ 2e-22 (no NaN).
        # For EMD (epsilon=0): pen=100 >> any legitimate cost.
        if epsilon > 0:
            penalty = max(2.0, 50.0 * epsilon)
        else:
            penalty = 100.0
        M_masked = M_kpg.copy()
        M_masked[Mask == 0] = penalty
        return solve_ot(a, b, M_masked, ot_type, epsilon, tau, adap_mass)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(args):
    eta1     = args.eta1
    eta2     = args.eta2
    eta3     = args.eta3
    tau      = args.tau
    epsilon  = args.epsilon
    mass     = args.mass
    k        = args.k
    ot_type  = args.ot_type
    use_kpg  = args.use_kpg
    alpha    = args.alpha
    tau_s    = args.tau_s
    tau_t    = args.tau_t
    n_shared_classes = args.n_shared_classes

    log_str = (
        "-" * 50 + "\n"
        " eta1 = {:.3f}, eta2 = {:.3f}, eta3 = {:.3f}\n"
        " ot_type = {}, epsilon = {:.4f}, tau = {:.4f}, mass = {:.3f}\n"
        " use_kpg = {}, alpha = {:.3f}, tau_s = {:.3f}, tau_t = {:.3f}\n"
        " n_shared_classes = {} / class_num = {}\n"
    ).format(eta1, eta2, eta3, ot_type, epsilon, tau, mass,
             use_kpg, alpha, tau_s, tau_t,
             n_shared_classes, args.class_num)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    train_bs, test_bs = args.batch_size, args.batch_size * 2

    dsets = {}
    dsets["source"] = data_list.ImageList(
        open(args.s_dset_path).readlines(), transform=image_train()
    )
    dsets["target"] = data_list.ImageList(
        open(args.t_dset_path).readlines(), transform=image_train()
    )
    dsets["test"] = data_list.ImageList(
        open(args.t_dset_path).readlines(), transform=image_test()
    )

    dset_loaders = {}

    # Balanced sampler for source: ensures each class appears every batch
    source_labels = torch.zeros(len(dsets["source"]))
    for i, line in enumerate(open(args.s_dset_path).readlines()):
        source_labels[i] = int(line.split()[1])
    train_batch_sampler = BalancedBatchSampler(source_labels, batch_size=train_bs)
    dset_loaders["source"] = DataLoader(
        dsets["source"], batch_sampler=train_batch_sampler, num_workers=args.worker
    )

    dset_loaders["target"] = DataLoader(
        dsets["target"], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True,
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker
    )

    if "ResNet" in args.net:
        params = {
            "resnet_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": 256,
            "new_cls": True,
            "class_num": args.class_num,
        }
        base_network = network.ResNetFc(**params)
    elif "VGG" in args.net:
        params = {
            "vgg_name": args.net,
            "use_bottleneck": True,
            "bottleneck_dim": 256,
            "new_cls": True,
            "class_num": args.class_num,
        }
        base_network = network.VGGFc(**params)

    base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()
    base_network = torch.nn.DataParallel(base_network).cuda()

    optimizer_config = {
        "type": torch.optim.SGD,
        "optim_params": {
            "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True
        },
        "lr_type": "inv",
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75},
    }
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    iter_source = iter(dset_loaders["source"])
    iter_target = iter(dset_loaders["target"])
    best_acc  = 0.0
    best_iter = 0

    for i in tqdm(range(args.max_iterations + 1)):

        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            base_network.train(False)
            temp_acc = image_classification(dset_loaders, base_network)
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)
            if best_acc < temp_acc:
                best_acc  = temp_acc
                best_iter = i

        if i % args.test_interval == 0:
            log_str = "\n{}, iter: {:05d}, source/target: {:02d}/{:02d}\n".format(
                args.name, i, train_bs, train_bs
            )
            args.out_file.write(log_str)
            args.out_file.flush()
            print(log_str)

        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # Adaptive mass ramp: linearly increase from 0 to `mass` over first half
        if i <= (args.max_iterations / 2):
            adap_mass = min(mass / (args.max_iterations / 2) * i + 0.01, mass)
        else:
            adap_mass = mass

        for _ in range(k):
            try:
                xs, ys = next(iter_source)
                xt, _  = next(iter_target)
            except StopIteration:
                iter_source = iter(dset_loaders["source"])
                iter_target = iter(dset_loaders["target"])
                xs, ys = next(iter_source)
                xt, _  = next(iter_target)

            xs, xt, ys = xs.cuda(), xt.cuda(), ys.cuda()
            g_xs, f_g_xs = base_network(xs)
            g_xt, f_g_xt = base_network(xt)

            pred_xt = F.softmax(f_g_xt, 1)

            classifier_loss = torch.nn.CrossEntropyLoss()(f_g_xs, ys) / k

            ys_oh   = F.one_hot(ys, num_classes=args.class_num).float()
            M_embed = torch.cdist(g_xs, g_xt) ** 2
            M_sce   = -torch.mm(ys_oh, torch.log(pred_xt + 1e-10).T)
            M       = eta1 * M_embed + eta2 * M_sce

            a      = ot.unif(g_xs.size(0))
            b      = ot.unif(g_xt.size(0))
            M_cpu  = M.detach().cpu().numpy()

            if use_kpg:
                pi = solve_ot_kpg(
                    a, b, M_cpu, ot_type, epsilon, tau, adap_mass,
                    feat_s=g_xs.detach().cpu().numpy(),
                    feat_t=g_xt.detach().cpu().numpy(),
                    ys_np=ys.cpu().numpy(),
                    pred_xt_np=pred_xt.detach().cpu().numpy(),
                    class_num=args.class_num,
                    tau_s=tau_s, tau_t=tau_t, alpha=alpha,
                    n_shared_classes=n_shared_classes,
                )
            else:
                M_norm = M_cpu / (M_cpu.max() + 1e-8)
                pi = solve_ot(a, b, M_norm, ot_type, epsilon, tau, adap_mass)

            pi = torch.from_numpy(pi).float().cuda()
            transfer_loss = eta3 * torch.sum(pi * M) / k

            if i % 100 == 0:
                log_str = (
                    "sum(pi)={:.4f}, transfer={:.4f}, "
                    "adap_mass={:.3f}, kps={}\n"
                ).format(
                    torch.sum(pi).item(), transfer_loss.item(), adap_mass,
                    len(select_keypoints_from_batch(
                        ys.cpu().numpy(), pred_xt.detach().cpu().numpy(),
                        args.class_num, n_shared_classes=n_shared_classes,
                    )[0]) if use_kpg else "N/A"
                )
                args.out_file.write(log_str)
                args.out_file.flush()
                print(log_str)

            total_loss = classifier_loss + transfer_loss
            total_loss.backward()

        optimizer.step()

    log_str = "Acc: {:.2f}\n".format(np.round(best_acc * 100, 2))
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    result_tag = "kpg_" + ot_type if use_kpg else ot_type
    with open(f"result_m{result_tag}.txt", "a") as f:
        f.write(
            "method {}, iter: {:05d}, precision: {:.5f}\n".format(
                args.name + "_" + args.output, best_iter, best_acc
            )
        )

    return best_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Mini-batch Keypoint-Guided OT for Partial Domain Adaptation"
    )
    parser.add_argument("--gpu_id",   type=str, nargs="?", default="0")
    parser.add_argument("--s",        type=int, default=0,  help="source domain index")
    parser.add_argument("--t",        type=int, default=1,  help="target domain index")
    parser.add_argument("--output",   type=str, default="run")
    parser.add_argument("--seed",     type=int, default=2020)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--batch_size",     type=int, default=65)
    parser.add_argument("--worker",         type=int, default=4)
    parser.add_argument("--net",      type=str, default="ResNet50",
                        choices=["ResNet50", "VGG16"])
    parser.add_argument("--dset",     type=str, default="office_home",
                        choices=["office_home"])
    parser.add_argument("--test_interval", type=int, default=500)
    parser.add_argument("--lr",       type=float, default=0.001)

    # OT parameters
    parser.add_argument("--ot_type",  type=str, default="pot",
                        choices=["ot", "uot", "pot"])
    parser.add_argument("--eta1",     type=float, default=0.003)
    parser.add_argument("--eta2",     type=float, default=0.75)
    parser.add_argument("--eta3",     type=float, default=10.0)
    parser.add_argument("--epsilon",  type=float, default=0.0)
    parser.add_argument("--tau",      type=float, default=0.06)
    parser.add_argument("--mass",     type=float, default=0.5)
    parser.add_argument("--k",        type=int,   default=1)

    # KPG-RL parameters
    parser.add_argument("--use_kpg",  action="store_true",
                        help="enable KPG-RL keypoint-guided OT")
    parser.add_argument("--alpha",    type=float, default=0.9,
                        help="cost = alpha*C_norm + (1-alpha)*G")
    parser.add_argument("--tau_s",    type=float, default=0.5,
                        help="source softmax temperature for relation profiles")
    parser.add_argument("--tau_t",    type=float, default=0.5,
                        help="target softmax temperature for relation profiles")
    parser.add_argument("--n_shared_classes", type=int, default=None,
                        help="restrict KPG keypoint candidate classes to "
                             "0..n_shared_classes-1.  Required for PDA "
                             "(set to 25 for Office-Home PDA).  None means "
                             "consider all class_num classes (closed-set DA).")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dset == "office_home":
        names = ["Art", "Clipart", "Product", "RealWorld"]
        args.class_num   = 65
        args.max_iterations = 5000
        args.test_interval  = 500

    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    args.s_dset_path = os.path.join(data_folder, args.dset, names[args.s] + "_list.txt")
    args.t_dset_path = os.path.join(data_folder, args.dset, names[args.t] + "_25_list.txt")

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join("snapshot", args.name, args.output)
    os.makedirs(args.output_dir, exist_ok=True)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    args.out_file.write(str(args) + "\n")
    args.out_file.flush()

    train(args)
