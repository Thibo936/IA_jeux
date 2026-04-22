# trainer.py — Boucle d'entraînement AlphaZero pour Hex 11×11
# Usage : python trainer.py [--iterations N] [--games G] [--simulations S] [--device cuda]

import argparse
import os
import sys
import time

def _barre(done: int, total: int, width: int = 28) -> str:
    """Retourne une barre de progression ASCII."""
    filled = int(width * done / total) if total > 0 else 0
    return '█' * filled + '░' * (width - filled)

import numpy as np
import torch
import torch.nn.functional as F

from network import HexNet
from mcts_az import MCTSAgent
from self_play import ReplayBuffer, run_self_play
from evaluate import evaluate_models
from config import (
    BATCH_SIZE, LEARNING_RATE, LR_ETA_MIN, WEIGHT_DECAY, TRAIN_STEPS,
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
    t_start = time.time()

    for step in range(steps):
        if len(buffer) < batch_size:
            break
        states, policies, values = buffer.sample(batch_size)
        l, lv, lp = train_step(net, optimizer, states, policies, values, device)
        losses.append(l)
        losses_v.append(lv)
        losses_p.append(lp)

        if scheduler is not None:
            scheduler.step()

        done = step + 1
        elapsed = time.time() - t_start
        vitesse = done / elapsed if elapsed > 0 else 0
        eta = (steps - done) / vitesse if vitesse > 0 else 0
        sys.stdout.write(
            f"\r  [{_barre(done, steps)}] {done}/{steps} | "
            f"loss={l:.4f} v={lv:.4f} p={lp:.4f} | "
            f"{vitesse:.1f}it/s | ETA:{eta:.0f}s"
        )
        sys.stdout.flush()

    sys.stdout.write('\n')

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
        state_dict = torch.load(path, map_location=device)
        try:
            net.load_state_dict(state_dict)
            print(f"  Checkpoint chargé : {path}")
            return True
        except RuntimeError:
            print(f"  Checkpoint incompatible ({path}), architecture changée → nouveau modèle.")
            return False
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
            "Installe une version PyTorch compatible GPU, ou lance avec --device auto / --device cpu."
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iterations * args.steps, eta_min=LR_ETA_MIN
    )

    # ─── Buffer ───────────────────────────────────────────────────────────────
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    buffer_path = os.path.join(CHECKPOINT_DIR, "replay_buffer.npz")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    n_loaded = buffer.load(buffer_path)
    if n_loaded > 0:
        print(f"  Buffer restauré : {n_loaded} exemples depuis {buffer_path}")

    # ─── Boucle principale AlphaZero ──────────────────────────────────────────
    t_start = time.time()
    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"Itération {iteration}/{args.iterations}")
        t0 = time.time()

        # 1. Self-play
        print(f"\n[1/3] Self-play ({args.games} parties, {args.simulations} sims)...")
        t_phase = time.time()
        agent = MCTSAgent(net, device=device, sims=args.simulations, add_dirichlet=True)
        stats = run_self_play(agent, buffer, num_games=args.games)
        t_sp = time.time() - t_phase
        print(f"  Blue : {stats['blue_wins']} | Red : {stats['red_wins']} | "
              f"Buffer : {len(buffer)} | Durée : {t_sp:.1f}s")

        if len(buffer) < args.batch:
            print(f"  Buffer trop petit ({len(buffer)} < {args.batch}), on attend la prochaine itération.")
            # Sauvegarde quand même pour ne pas perdre les données self-play accumulées
            save_checkpoint(net, os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt"))
            if not os.path.isfile(BEST_MODEL_FILE):
                save_checkpoint(net, BEST_MODEL_FILE)
            continue

        # 2. Entraînement
        print(f"\n[2/3] Entraînement ({args.steps} steps, batch={args.batch})...")
        t_phase = time.time()
        metrics = train_epoch(net, optimizer, scheduler, buffer, args.steps, args.batch, device)
        t_train = time.time() - t_phase
        print(f"  Loss totale={metrics['loss']:.4f}  valeur={metrics['loss_value']:.4f}  "
              f"politique={metrics['loss_policy']:.4f} | Durée : {t_train:.1f}s")

        # Sauvegarde intermédiaire
        iter_path = os.path.join(CHECKPOINT_DIR, f"model_iter_{iteration:04d}.pt")
        save_checkpoint(net, iter_path)

        # 3. Évaluation (nouveau vs meilleur)
        if not args.no_eval:
            print(f"\n[3/3] Évaluation ({args.eval_games} parties)...")
            t_phase = time.time()
            # Charger l'ancien meilleur modèle
            best_net = HexNet().to(device)
            if not load_checkpoint(best_net, BEST_MODEL_FILE, device):
                # Pas encore de meilleur modèle → le nouveau devient le meilleur
                save_checkpoint(net, BEST_MODEL_FILE)
            else:
                win_rate = evaluate_models(net, best_net, device, num_games=args.eval_games)
                t_eval = time.time() - t_phase
                print(f"  Win rate nouveau modèle : {win_rate:.1%} (seuil={WIN_RATE_THRESHOLD:.0%}) | "
                      f"Durée : {t_eval:.1f}s")
                if win_rate >= WIN_RATE_THRESHOLD:
                    print("  ✓ Nouveau modèle accepté !")
                    save_checkpoint(net, BEST_MODEL_FILE)
                else:
                    print("  ✗ Ancien modèle conservé.")
                    # Recharger l'ancien meilleur pour continuer l'entraînement
                    load_checkpoint(net, BEST_MODEL_FILE, device)
                    # Réinitialiser l'optimiseur (les moments Adam sont obsolètes)
                    remaining_steps = (args.iterations - iteration) * args.steps
                    optimizer = torch.optim.Adam(
                        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(remaining_steps, 1), eta_min=LR_ETA_MIN
                    )

        # Sauvegarde du buffer
        buffer.save(buffer_path)

        t1 = time.time()
        t_iter = t1 - t0
        # ETA global
        iters_done = iteration
        iters_left = args.iterations - iters_done
        t_total_elapsed = t1 - t_start
        eta_global = (t_total_elapsed / iters_done) * iters_left if iters_done > 0 else 0
        h, m = divmod(int(eta_global), 3600)
        m, s = divmod(m, 60)
        print(f"\n  Durée itération : {t_iter:.1f}s | "
              f"Temps total : {t_total_elapsed/60:.1f}min | "
              f"ETA fin : {h:02d}h{m:02d}m{s:02d}s")

    t_total = time.time() - t_start
    print("\nEntraînement terminé.")
    print(f"Meilleur modèle : {BEST_MODEL_FILE}")
    print(f"Durée totale : {t_total/3600:.0f}h {(t_total%3600)/60:.0f}m {t_total%60:.0f}s")


if __name__ == "__main__":
    main()
