# trainer.py — Boucle d'entraînement AlphaZero pour Hex 11×11
# Usage : python trainer.py [--iterations N] [--games G] [--simulations S] [--device cuda]

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from network import HexNet
from mcts_az import MCTSAgent
from self_play import ReplayBuffer, run_self_play
from evaluate import evaluate_models
from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, TRAIN_STEPS,
    GAMES_PER_ITER, MCTS_SIMULATIONS, CHECKPOINT_DIR, BEST_MODEL_FILE,
    REPLAY_BUFFER_SIZE, WIN_RATE_THRESHOLD, EVAL_GAMES,
)


def train_step(
    net: HexNet,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Un pas d'entraînement sur un batch.
    Retourne (loss_totale, loss_valeur, loss_politique).
    """
    net.train()
    states_t   = torch.from_numpy(states).to(device)
    policies_t = torch.from_numpy(policies).to(device)
    values_t   = torch.from_numpy(values).unsqueeze(1).to(device)

    log_p, v = net(states_t)

    # Loss politique : cross-entropie entre distribution MCTS et sortie réseau
    loss_policy = -(policies_t * log_p).sum(dim=1).mean()

    # Loss valeur : MSE entre valeur réseau et résultat de la partie
    loss_value = F.mse_loss(v, values_t)

    loss = loss_policy + loss_value

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_value.item(), loss_policy.item()


def train_epoch(
    net: HexNet,
    optimizer: torch.optim.Optimizer,
    scheduler,
    buffer: ReplayBuffer,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Lance `steps` pas d'entraînement. Retourne les métriques moyennes."""
    losses, losses_v, losses_p = [], [], []

    for _ in range(steps):
        if len(buffer) < batch_size:
            break
        states, policies, values = buffer.sample(batch_size)
        l, lv, lp = train_step(net, optimizer, states, policies, values, device)
        losses.append(l)
        losses_v.append(lv)
        losses_p.append(lp)

    if scheduler is not None:
        scheduler.step()

    return {
        'loss':        float(np.mean(losses))   if losses   else 0.0,
        'loss_value':  float(np.mean(losses_v)) if losses_v else 0.0,
        'loss_policy': float(np.mean(losses_p)) if losses_p else 0.0,
    }


def save_checkpoint(net: HexNet, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(net.state_dict(), path)
    print(f"  Checkpoint sauvegardé : {path}")


def load_checkpoint(net: HexNet, path: str, device: torch.device) -> bool:
    if os.path.isfile(path):
        net.load_state_dict(torch.load(path, map_location=device))
        print(f"  Checkpoint chargé : {path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Entraînement AlphaZero Hex 11×11")
    parser.add_argument("--iterations",   type=int, default=20,             help="Nombre d'itérations AlphaZero")
    parser.add_argument("--games",        type=int, default=GAMES_PER_ITER, help="Parties par itération")
    parser.add_argument("--simulations",  type=int, default=MCTS_SIMULATIONS, help="Simulations MCTS par coup")
    parser.add_argument("--steps",        type=int, default=TRAIN_STEPS,    help="Steps d'entraînement par iter")
    parser.add_argument("--batch",        type=int, default=BATCH_SIZE,     help="Batch size")
    parser.add_argument("--device",       type=str, default="auto",         help="cuda / cpu / auto")
    parser.add_argument("--eval-games",   type=int, default=EVAL_GAMES,     help="Parties d'évaluation")
    parser.add_argument("--no-eval",      action="store_true",              help="Désactive l'évaluation")
    args = parser.parse_args()

    # ─── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            raise SystemExit(
                "Erreur : PyTorch ROCm est installé, mais aucun GPU n'est détecté. "
                "Vérifie l'installation ROCm et les variables d'environnement GPU."
            )
        raise SystemExit(
            "Erreur : aucun GPU CUDA/ROCm n'est disponible. "
            "Installe une version PyTorch compatible GPU ou lance avec --device cpu."
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Device : {device} ({gpu_name})")
        else:
            print(f"Device : {device}")
    else:
        print(f"Device : {device}")

    # ─── Réseau ───────────────────────────────────────────────────────────────
    net = HexNet().to(device)
    if not load_checkpoint(net, BEST_MODEL_FILE, device):
        print("  Nouveau modèle initialisé (aucun checkpoint trouvé).")

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ─── Buffer ───────────────────────────────────────────────────────────────
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.npz")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    n_loaded = buffer.load(buffer_path)
    if n_loaded > 0:
        print(f"  Buffer restauré : {n_loaded} exemples depuis {buffer_path}")

    # ─── Boucle principale AlphaZero ──────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"Itération {iteration}/{args.iterations}")
        t0 = time.time()

        # 1. Self-play
        print(f"\n[1/3] Self-play ({args.games} parties, {args.simulations} sims)...")
        agent = MCTSAgent(net, device=device, sims=args.simulations, add_dirichlet=True)
        stats = run_self_play(agent, buffer, num_games=args.games)
        print(f"  Blue : {stats['blue_wins']} | Red : {stats['red_wins']} | Buffer : {len(buffer)}")

        if len(buffer) < args.batch:
            print(f"  Buffer trop petit ({len(buffer)} < {args.batch}), on attend la prochaine itération.")
            # Sauvegarde quand même pour ne pas perdre les données self-play accumulées
            save_checkpoint(net, os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt"))
            if not os.path.isfile(BEST_MODEL_FILE):
                save_checkpoint(net, BEST_MODEL_FILE)
            continue

        # 2. Entraînement
        print(f"\n[2/3] Entraînement ({args.steps} steps, batch={args.batch})...")
        metrics = train_epoch(net, optimizer, None, buffer, args.steps, args.batch, device)
        print(f"  Loss totale={metrics['loss']:.4f}  valeur={metrics['loss_value']:.4f}  politique={metrics['loss_policy']:.4f}")

        # Sauvegarde intermédiaire
        iter_path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt")
        save_checkpoint(net, iter_path)

        # 3. Évaluation (nouveau vs meilleur)
        if not args.no_eval:
            print(f"\n[3/3] Évaluation ({args.eval_games} parties)...")
            # Charger l'ancien meilleur modèle
            best_net = HexNet().to(device)
            if not load_checkpoint(best_net, BEST_MODEL_FILE, device):
                # Pas encore de meilleur modèle → le nouveau devient le meilleur
                save_checkpoint(net, BEST_MODEL_FILE)
            else:
                win_rate = evaluate_models(net, best_net, device, num_games=args.eval_games)
                print(f"  Win rate nouveau modèle : {win_rate:.1%} (seuil={WIN_RATE_THRESHOLD:.0%})")
                if win_rate >= WIN_RATE_THRESHOLD:
                    print("  ✓ Nouveau modèle accepté !")
                    save_checkpoint(net, BEST_MODEL_FILE)
                else:
                    print("  ✗ Ancien modèle conservé.")
                    # Recharger l'ancien meilleur pour continuer l'entraînement
                    load_checkpoint(net, BEST_MODEL_FILE, device)

        # Sauvegarde du buffer
        buffer.save(buffer_path)

        t1 = time.time()
        print(f"\n  Durée itération : {t1-t0:.1f}s")

    print("\nEntraînement terminé.")
    print(f"Meilleur modèle : {BEST_MODEL_FILE}")


if __name__ == "__main__":
    main()
