from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from tqdm import trange

from .data import EpisodeSampler
from .metrics import accuracy, summarize_accuracies
from .semantics import SemanticBank


def move_episode(episode, device: torch.device):
    episode.support_images = episode.support_images.to(device)
    episode.support_labels = episode.support_labels.to(device)
    episode.query_images = episode.query_images.to(device)
    episode.query_labels = episode.query_labels.to(device)
    return episode


def train_semfew(
    model: nn.Module,
    sampler: EpisodeSampler,
    val_sampler: EpisodeSampler | None,
    semantic_bank: SemanticBank,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    episodes = int(config.get("train_episodes", 1000))
    val_interval = int(config.get("val_interval", 100))
    val_episodes = int(config.get("val_episodes", 100))
    best_acc = -1.0
    history = []

    progress = trange(1, episodes + 1, desc="training", dynamic_ncols=True)
    for step in progress:
        model.train()
        episode = move_episode(sampler.sample(), device)
        semantic_centers = semantic_bank.encode(episode.class_names, device=device)
        output = model(episode.support_images, episode.support_labels, episode.query_images, semantic_centers)
        loss = criterion(output.logits, episode.query_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(output.logits.detach(), episode.query_labels)
        record = {"episode": step, "loss": float(loss.item()), "train_accuracy": train_acc}

        if val_sampler is not None and (step % val_interval == 0 or step == episodes):
            stats = evaluate_semfew(model, val_sampler, semantic_bank, val_episodes, device)
            record["val_accuracy"] = stats.mean_accuracy
            record["val_ci95"] = stats.ci95
            if stats.mean_accuracy > best_acc:
                best_acc = stats.mean_accuracy
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_episode": step,
                        "best_accuracy": best_acc,
                    },
                    output_dir / "best_model.pt",
                )
        history.append(record)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_acc:.3f}", best=f"{max(best_acc, 0):.3f}")

    if best_acc < 0:
        torch.save({"model_state_dict": model.state_dict(), "config": config}, output_dir / "best_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_dir / "last_model.pt")
    summary = {"best_accuracy": max(best_acc, 0.0), "history": history, "config": config}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


@torch.no_grad()
def evaluate_semfew(
    model: nn.Module,
    sampler: EpisodeSampler,
    semantic_bank: SemanticBank,
    episodes: int,
    device: torch.device,
):
    model.eval()
    accuracies = []
    for _ in range(episodes):
        episode = move_episode(sampler.sample(), device)
        semantic_centers = semantic_bank.encode(episode.class_names, device=device)
        output = model(episode.support_images, episode.support_labels, episode.query_images, semantic_centers)
        accuracies.append(accuracy(output.logits, episode.query_labels))
    return summarize_accuracies(accuracies)

