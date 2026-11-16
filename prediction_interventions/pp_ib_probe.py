"""
pp_ib_probe.py  —  单文件诊断+出图工具，不依赖旧的 gate_diagnostic.py

用途：在 µ 被稳住（调小 kl_weight）的前提下，验证 gate 是否能学出*真实*双峰，
并产出 Fig.4(a) 所需的逐-epoch 演化数据。一次 5-epoch 短跑即可。

集成（trainer 里 3 处，见文件末尾 INTEGRATION 注释）：
  1) 训练开始前： probe = PPIBProbe(out_dir='results/pp_ib_fix')
  2) 每个 step（forward 后、step() 前）： probe.log_step(mu=..., gate=..., gate_logit=..., mask=...)
  3) 每个 epoch 末： probe.epoch_end(epoch)
  4) 训练结束： probe.finalize()   # 存 npy + 画 Fig.4(a) + 打印健康判定

依赖：numpy, torch, matplotlib
"""
import json
import os

import numpy as np
import torch


def _to_np(x, mask=None):
    """把 (B, L) 或 (B, L, D) 的 tensor 摊平成 1D，mask=(B,L) 时只取有效位置。"""
    x = x.detach().float().cpu()
    if x.dim() == 3:
        x = x.mean(-1)
    x = x.reshape(x.shape[0], -1) if x.dim() == 2 else x.reshape(1, -1)
    if mask is not None:
        m = mask.detach().bool().cpu().reshape(x.shape)
        return x[m].numpy().ravel()
    return x.numpy().ravel()


def _mu_norm(mu, mask=None):
    """每个 token 的 ||µ|| 的均值。mu: (B,L,D)。"""
    mu = mu.detach().float().cpu()
    n = mu.norm(dim=-1)
    if mask is not None:
        m = mask.detach().bool().cpu()
        return n[m].mean().item()
    return n.mean().item()


class PPIBProbe:
    def __init__(
        self,
        out_dir="results/pp_ib_fix",
        beta_polar: float = 0.01,
        lambda_balance: float = 1.0,
    ):
        self.out_dir = out_dir
        self.beta_polar = float(beta_polar)
        self.lambda_balance = float(lambda_balance)
        os.makedirs(out_dir, exist_ok=True)
        self._reset_acc()
        self.history = []
        self.gate_hists = []

    def _reset_acc(self):
        self._gates = []
        self._logit_std = []
        self._mu_norms = []

    @torch.no_grad()
    def log_step(self, mu, gate, gate_logit, mask=None):
        """每个 step 调一次。gate/gate_logit: (B,L) 或 (B,L,1)。mu: (B,L,D)。mask: (B,L)。"""
        g = _to_np(gate, mask)
        self._gates.append(g)
        self._logit_std.append(float(np.std(_to_np(gate_logit, mask))))
        self._mu_norms.append(_mu_norm(mu, mask))

    @torch.no_grad()
    def epoch_end(self, epoch):
        g = np.concatenate(self._gates) if self._gates else np.array([0.5])
        gate_mean = float(g.mean())
        gate_std = float(g.std())
        frac_extreme = float(np.mean((g < 0.15) | (g > 0.85)))
        frac_mid = float(np.mean((g > 0.35) & (g < 0.65)))
        frac_low = float(np.mean(g < 0.15))
        frac_high = float(np.mean(g > 0.85))
        true_bimodal = (frac_low > 0.15) and (frac_high > 0.15)
        mu_norm = float(np.mean(self._mu_norms))
        balance_term = float(max(abs(gate_mean - 0.5) - 0.2, 0.0) ** 2)
        eff_balance = self.beta_polar * self.lambda_balance * balance_term

        rec = dict(
            epoch=epoch,
            gate_mean=gate_mean,
            gate_std=gate_std,
            frac_extreme=frac_extreme,
            frac_mid=frac_mid,
            frac_low=frac_low,
            frac_high=frac_high,
            true_bimodal=bool(true_bimodal),
            logit_std=float(np.mean(self._logit_std)),
            mu_norm=mu_norm,
            balance_term=balance_term,
            eff_balance=eff_balance,
        )
        self.history.append(rec)
        samp = g if g.size <= 4000 else g[np.random.choice(g.size, 4000, replace=False)]
        self.gate_hists.append(samp.astype(np.float32))

        print(
            f"[probe] ep{epoch}: g_mean={gate_mean:.3f} "
            f"low={frac_low:.2f} high={frac_high:.2f} mid={frac_mid:.2f} "
            f"true_bimodal={true_bimodal} ||mu||={mu_norm:.3f} "
            f"B={balance_term:.4f} eff_bal={eff_balance:.6f}",
            flush=True,
        )
        self._reset_acc()

    def finalize(self):
        np.save(
            os.path.join(self.out_dir, "gate_evolution.npy"),
            np.array(self.gate_hists, dtype=object),
            allow_pickle=True,
        )
        with open(os.path.join(self.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        self._render_fig4a()
        return self._verdict()

    def _verdict(self):
        h = self.history
        if len(h) < 2:
            print(">> VERDICT: INSUFFICIENT_DATA", flush=True)
            return "INSUFFICIENT_DATA"
        mu0, mu1 = h[0]["mu_norm"], h[-1]["mu_norm"]
        mu_stable = mu1 > 0.6 * mu0
        bimodal_last = h[-1]["true_bimodal"]
        learned = h[-1]["frac_extreme"] > h[0]["frac_extreme"] + 0.05

        print("\n================ VERDICT ================", flush=True)
        print(f"||mu||: {mu0:.3f} -> {mu1:.3f}  (stable={mu_stable})", flush=True)
        print(
            f"true_bimodal(last)={bimodal_last}  learned_over_epochs={learned}",
            flush=True,
        )
        if mu_stable and bimodal_last and learned:
            tag = "HEALTHY"
            print(">> HEALTHY: µ 稳住 + gate 学出真双峰。", flush=True)
            print(">> 这套配置可直接拿去重扫 Fig.3 的 β。Fig.4(a) 数据已可用。", flush=True)
        elif bimodal_last and not mu_stable:
            tag = "FAKE_BIMODAL"
            print(">> FAKE_BIMODAL: gate 极化了但 µ 仍在塌——gate 在控制空表示。", flush=True)
            print(">> 继续调小 kl_weight（再降一档）或对 µ 单独降权后重跑。", flush=True)
        elif mu_stable and not bimodal_last:
            tag = "STUCK_NEED_BETA"
            print(">> STUCK_NEED_BETA: µ 稳了但没极化：β_polar 太弱。把 β_polar 升到 0.1 再跑。", flush=True)
        else:
            tag = "NEED_KL_AND_BETA"
            print(">> NEED_KL_AND_BETA: 两者都未达标：先保 µ（降 kl_weight），再调 β_polar。", flush=True)
        print("=========================================\n", flush=True)
        return tag

    def _render_fig4a(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = self.history
        epochs = [r["epoch"] for r in h]
        fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2), dpi=300)

        ax[0].hist(
            self.gate_hists[0],
            bins=30,
            range=(0, 1),
            alpha=0.55,
            label=f"epoch {epochs[0]}",
            color="#9aa0a6",
        )
        ax[0].hist(
            self.gate_hists[-1],
            bins=30,
            range=(0, 1),
            alpha=0.75,
            label=f"epoch {epochs[-1]}",
            color="#1a73e8",
        )
        ax[0].set_xlabel("gate value $g$")
        ax[0].set_ylabel("count")
        ax[0].set_title("(a) gate distribution", fontsize=8)
        ax[0].legend(fontsize=6)

        fe = [r["frac_extreme"] for r in h]
        ax[1].plot(epochs, fe, "o-", color="#1a73e8", ms=3)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("epoch")
        ax[1].set_title("(b) frac. near {0,1}", fontsize=8)

        mn = [r["mu_norm"] for r in h]
        ax[2].plot(epochs, mn, "s-", color="#34a853", ms=3)
        ax[2].set_xlabel("epoch")
        ax[2].set_title(r"(c) $\|\mu\|$ (must stay stable)", fontsize=8)

        for a in ax:
            a.tick_params(labelsize=6)
        plt.tight_layout()
        out = os.path.join(self.out_dir, "fig4a_evolution.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"[probe] Fig.4(a) saved -> {out}", flush=True)
