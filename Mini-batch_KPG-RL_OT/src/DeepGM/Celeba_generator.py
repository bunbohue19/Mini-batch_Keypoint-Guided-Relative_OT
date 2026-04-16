"""CelebA generator + discriminator with optional KPG-RL guided OT.

Architecture unchanged from baseline (64x64 images, one extra conv layer
compared to CIFAR).  OT solver calls are routed through _solve() which
dispatches to KPG-RL when use_kpg=True.
"""

import numpy as np
import ot
import torch
import torch.nn as nn
from kpg_ot import solve_ot, solve_ot_kpg
from utils import sliced_wasserstein_distance


class Discriminator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(Discriminator, self).__init__()
        self.main1 = nn.Sequential(
            nn.Conv2d(num_chanel, hidden_chanels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_chanels, hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_chanels * 2, hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_chanels * 4, hidden_chanels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels * 8),
            nn.Tanh(),
        )
        self.main2 = nn.Sequential(
            nn.Conv2d(hidden_chanels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.main1(x)
        y = self.main2(h).view(x.shape[0], -1)
        return y, h


class Generator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, hidden_chanels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_chanels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_chanels * 8, hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_chanels * 4, hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_chanels * 2, hidden_chanels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_chanels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_chanels, num_chanel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(z.shape[0], self.latent_size, 1, 1))


class Celeba_Generator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels, device,
                 use_kpg=False, n_kp=5, alpha=0.5, tau_s=0.1, tau_t=0.1):
        super(Celeba_Generator, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels = hidden_chanels
        self.device = device
        self.decoder = Generator(image_size, latent_size, num_chanel, hidden_chanels)
        self.use_kpg = use_kpg
        self.n_kp = n_kp
        self.alpha = alpha
        self.tau_s = tau_s
        self.tau_t = tau_t

    def _solve(self, cost_matrix, method, reg, tau, mass, feat_real=None, feat_fake=None):
        a, b = ot.unif(cost_matrix.size(0)), ot.unif(cost_matrix.size(1))
        C_np = cost_matrix.detach().cpu().numpy()
        if self.use_kpg and feat_real is not None and feat_fake is not None:
            return torch.from_numpy(solve_ot_kpg(
                a, b, C_np, method, reg, tau, mass,
                feat_real=feat_real, feat_fake=feat_fake,
                n_kp=self.n_kp, alpha=self.alpha,
                tau_s=self.tau_s, tau_t=self.tau_t,
            )).cuda(self.device)
        return torch.from_numpy(solve_ot(a, b, C_np, method, reg, tau, mass)).cuda(self.device)

    def _ot_block(self, fd, ff, cost, method, reg, tau, mass):
        feat_r = fd.detach().cpu().numpy()
        feat_f = ff.detach().cpu().numpy()
        pi = self._solve(cost, method, reg, tau, mass, feat_r, feat_f)
        return pi, torch.sum(pi * cost)

    def train_minibatch(
        self, model_op, discriminator, optimizer, data, k, m,
        method="OT", reg=0, breg=0, tau=1, mass=0.9, L=1000,
        bomb=False, ebomb=False,
    ):
        z = torch.randn((data.shape[0], self.latent_size))
        if (data.shape[0] % k) == 0:
            inds_data = np.split(np.arange(data.shape[0]), k)
            inds_z = np.split(np.arange(z.shape[0]), k)
        else:
            real_k = int(data.shape[0] / m)
            if real_k != 0:
                inds_data = list(np.split(np.arange(real_k * m), real_k))
                inds_z = list(np.split(np.arange(real_k * m), real_k))
                k = real_k
                if method != "sliced" and data.shape[0] % m != 0:
                    inds_data.append(np.arange(real_k * m, data.shape[0]))
                    inds_z.append(np.arange(real_k * m, data.shape[0]))
                    k += 1
            else:
                k = 1
                inds_data = [np.arange(data.shape[0])]
                inds_z = [np.arange(data.shape[0])]

        # ── Discriminator phase ──────────────────────────────────────────
        dloss = []
        if (bomb or ebomb) and method != "sliced":
            self.eval(); discriminator.eval()
            with torch.no_grad():
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                        fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                        cost = torch.cdist(fd, ff) ** 2
                        _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                        dloss.append(loss)
                big_C = torch.stack(dloss).view(k, k)
                plan = ot.emd([], [], big_C.detach().cpu().numpy()) if bomb else \
                       ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)

        Dloss = 0
        self.train(); discriminator.train()
        if method == "sliced":
            optimizer.zero_grad()
            for i in range(k):
                data_mb = data[inds_data[i]].to(self.device)
                y_data, _ = discriminator(data_mb)
                label = torch.full((data_mb.shape[0], 1), 1, dtype=torch.float32, device=self.device)
                (1.0 / (k**2) * nn.BCELoss(reduction="sum")(y_data, label)).backward()
            optimizer.step()
            optimizer.zero_grad()
            for j in range(k):
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                y_fake, _ = discriminator(fake_mb)
                label = torch.full((z_mb.shape[0], 1), 0, dtype=torch.float32, device=self.device)
                (1.0 / (k**2) * nn.BCELoss(reduction="sum")(y_fake, label)).backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            for i in range(k):
                for j in range(k):
                    if (bomb or ebomb) and plan[i, j] == 0:
                        continue
                    data_mb = data[inds_data[i]].to(self.device)
                    z_mb = z[inds_z[j]].cuda(self.device)
                    fake_mb = self.decoder(z_mb)
                    _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                    fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                    cost = torch.cdist(fd, ff) ** 2
                    _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                    w = plan[i, j] if (bomb or ebomb) else 1.0 / (k**2)
                    mloss = -w * loss
                    Dloss += mloss
                    mloss.backward()
            optimizer.step()

        # ── Generator phase ──────────────────────────────────────────────
        gloss = []
        if bomb or ebomb:
            with torch.no_grad():
                self.eval(); discriminator.eval()
                for i in range(k):
                    for j in range(k):
                        data_mb = data[inds_data[i]].to(self.device)
                        z_mb = z[inds_z[j]].cuda(self.device)
                        fake_mb = self.decoder(z_mb)
                        _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                        fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                        if method == "sliced":
                            gloss.append(sliced_wasserstein_distance(fd, ff, num_projections=L, device=self.device))
                        else:
                            cost = torch.cdist(fd, ff) ** 2
                            _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                            gloss.append(loss)
                big_C = torch.stack(gloss).view(k, k)
                plan = ot.emd([], [], big_C.detach().cpu().numpy()) if bomb else \
                       ot.sinkhorn([], [], big_C.detach().cpu().numpy(), reg=breg)

        self.train(); discriminator.train()
        model_op.zero_grad()
        G_loss = 0
        for i in range(k):
            for j in range(k):
                if (bomb or ebomb) and plan[i, j] == 0:
                    continue
                data_mb = data[inds_data[i]].to(self.device)
                z_mb = z[inds_z[j]].cuda(self.device)
                fake_mb = self.decoder(z_mb)
                _, fd = discriminator(data_mb); _, ff = discriminator(fake_mb)
                fd = fd.view(data_mb.size(0), -1); ff = ff.view(z_mb.size(0), -1)
                if method == "sliced":
                    loss = sliced_wasserstein_distance(fd, ff, num_projections=L, device=self.device)
                else:
                    cost = torch.cdist(fd, ff) ** 2
                    _, loss = self._ot_block(fd, ff, cost, method, reg, tau, mass)
                w = plan[i, j] if (bomb or ebomb) else 1.0 / (k**2)
                mloss = w * loss
                G_loss += mloss
                mloss.backward()
        model_op.step()
        return G_loss, Dloss
