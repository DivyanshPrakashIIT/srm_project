import argparse
from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
import yaml

from srm.data.dataset import create_dataloaders
from srm.models import build_model


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/model_m1.yaml") -> None:
    config = load_config(config_path)

    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 10)
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Build model & move to target device
    model = build_model(config).to(device)

    # Data loaders
    train_loader, val_loader = create_dataloaders(config_path)

    # Loss, Optimizer, Cosine Annealing Scheduler & AMP
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=train_cfg.get("min_lr", 1e-6)
    )

    use_amp = torch.cuda.is_available() and train_cfg.get("use_amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(lr_imgs)
                sr_imgs = out["sr"]
                loss = criterion(sr_imgs, hr_imgs)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * lr_imgs.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = model(lr_imgs)
                    sr_imgs = out["sr"]
                    loss = criterion(sr_imgs, hr_imgs)

                val_loss += loss.item() * lr_imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }

        torch.save(checkpoint, save_dir / "latest_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f" -> Saved new best checkpoint (Val Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SRM Super-Resolution Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_m1.yaml",
        help="Path to YAML training configuration",
    )
    args = parser.parse_args()
    train(args.config)