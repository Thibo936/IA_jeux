# network.py — Réseau de neurones CNN pour AlphaZero Hex 11×11
# Architecture : blocs résiduels (ResNet-style) + tête Politique + tête Valeur

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CELLS, NUM_CHANNELS, NUM_RES_BLOCKS, INPUT_CHANNELS


class ResBlock(nn.Module):
    """Bloc résiduel : deux Conv 3×3 + BN + connexion sautée."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexNet(nn.Module):
    """
    Réseau AlphaZero pour Hex 11×11.

    Entrée  : (batch, 3, 11, 11)
    Sorties :
      - politique : (batch, 121) — distribution sur les 121 coups (log_softmax)
      - valeur    : (batch, 1)   — estimation du résultat ∈ [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        channels: int = NUM_CHANNELS,
        num_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()

        # ─── Tronc commun ──────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # ─── Tête Politique ────────────────────────────────────────────────
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * NUM_CELLS, NUM_CELLS)

        # ─── Tête Valeur ───────────────────────────────────────────────────
        self.value_conv  = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(NUM_CELLS, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (batch, 3, 11, 11)
        Retourne (log_policy, value) :
          log_policy : (batch, 121) — log-probabilités (log_softmax)
          value      : (batch, 1)   — tanh ∈ [-1, 1]
        """
        # Tronc
        x = self.stem(x)
        x = self.res_blocks(x)

        # Tête Politique
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)          # (batch, 2*121)
        p = self.policy_fc(p)              # (batch, 121)
        log_p = F.log_softmax(p, dim=1)

        # Tête Valeur
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)          # (batch, 121)
        v = F.relu(self.value_fc1(v))      # (batch, 256)
        v = torch.tanh(self.value_fc2(v))  # (batch, 1)

        return log_p, v

    def predict(
        self,
        state_tensor: "np.ndarray",
        legal_mask: "np.ndarray",
        device: torch.device,
    ) -> tuple["np.ndarray", float]:
        """
        Inférence pour un seul état (pas de gradient).

        state_tensor : numpy (3, 11, 11)
        legal_mask   : numpy bool (121,)
        Retourne (policy_probs, value) :
          policy_probs : numpy float32 (121,) masquée et renormalisée sur coups légaux
          value        : float ∈ [-1, 1]
        """
        import numpy as np

        self.eval()
        with torch.no_grad(), torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            t = torch.from_numpy(state_tensor).unsqueeze(0).to(device)  # (1,3,11,11)
            log_p, v = self(t)
            log_p = log_p.float().squeeze(0).cpu().numpy()   # (121,)
            value  = v.float().squeeze().item()

        # Masquer les coups illégaux et renormaliser
        policy = np.exp(log_p)
        policy = policy * legal_mask.astype(np.float32)
        s = policy.sum()
        if s > 1e-8:
            policy /= s
        else:
            # Politique uniforme sur les coups légaux si le réseau est saturé
            policy = legal_mask.astype(np.float32)
            policy /= policy.sum()

        return policy, value

    def batch_predict(
        self,
        states: "np.ndarray",
        legal_masks: "np.ndarray",
        device: torch.device,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """
        Inférence batchée (pas de gradient).

        states      : numpy (B, 3, 11, 11)
        legal_masks : numpy bool (B, 121)
        Retourne (policies, values) :
          policies : numpy float32 (B, 121)
          values   : numpy float32 (B,)
        """
        import numpy as np

        self.eval()
        with torch.no_grad(), torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
            t = torch.from_numpy(states).to(device)
            log_p, v = self(t)
            log_p = log_p.float().cpu().numpy()   # (B, 121)
            values = v.float().squeeze(1).cpu().numpy()  # (B,)

        policies = np.exp(log_p)
        masks = legal_masks.astype(np.float32)
        policies *= masks
        sums = policies.sum(axis=1, keepdims=True)
        # Éviter division par zéro
        safe = sums > 1e-8
        policies = np.where(safe, policies / np.maximum(sums, 1e-8), masks / np.maximum(masks.sum(axis=1, keepdims=True), 1))

        return policies, values


# ─── Test rapide d'overfit ────────────────────────────────────────────────────

def test_overfit(device_str: str = "cpu", steps: int = 200):
    """
    Vérifie que le réseau peut mémoriser 10 positions synthétiques.
    Si la loss descend < 0.01, l'architecture est correcte.
    """
    import numpy as np

    device = torch.device(device_str)
    net = HexNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Données synthétiques
    np.random.seed(42)
    X = torch.randn(10, 3, 11, 11, device=device)
    P = torch.softmax(torch.randn(10, NUM_CELLS, device=device), dim=1)
    V = torch.tanh(torch.randn(10, 1, device=device))

    net.train()
    for step in range(steps):
        log_p, v = net(X)
        loss_p = -(P * log_p).sum(dim=1).mean()
        loss_v = F.mse_loss(v, V)
        loss   = loss_p + loss_v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}  loss={loss.item():.4f}")

    print(f"Overfit test terminé. Loss finale : {loss.item():.4f}")
    return loss.item() < 0.5


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    ok = test_overfit(device)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
