import logging

import hydra
import torch
import torchmetrics
import torchvision.transforms.v2 as T
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from tqdm.auto import tqdm

from model import SwinTransformer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    log_dir = HydraConfig.get().run.dir
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    metrics = torchmetrics.MetricCollection(
        {
            "f1": torchmetrics.F1Score(task="multiclass", num_classes=cfg.model.num_classes),
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=cfg.model.num_classes),
        }
    ).to(device)

    train_loss = torchmetrics.MeanMetric().to(device)
    valid_loss = torchmetrics.MeanMetric().to(device)

    train_metrics = metrics.clone(postfix="/train")
    valid_metrics = metrics.clone(postfix="/valid")

    logging.info(f"Computing metrics: {[metric for metric in metrics]}")

    epochs_without_improvement = 0
    best_valid_loss = torch.inf

    model = SwinTransformer(**cfg.model).to(device)
    model.compile(fullgraph=True)

    criterion = nn.CrossEntropyLoss(**cfg.criterion)
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)

    logging.info(f"Using model: {HydraConfig.get().runtime.choices.get('model')}")

    num_epochs = cfg.trainer.num_epochs
    warmup_steps = cfg.trainer.warmup_steps
    schedulers = [
        LinearLR(optimizer, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, num_epochs - warmup_steps),
    ]
    lr_scheduler = SequentialLR(optimizer, schedulers, [warmup_steps])

    toTensor = T.Compose((T.ToImage(), T.ToDtype(torch.float, scale=True), T.Resize((256, 256))))
    train_set = CIFAR100("data/", train=True, transform=toTensor, download=True)
    valid_set = CIFAR100("data/", train=False, transform=toTensor, download=True)
    train_loader = DataLoader(train_set, **cfg.data.train)
    valid_loader = DataLoader(valid_set, **cfg.data.valid)
    logging.info(f"Using {len(train_set)} training and {len(valid_set)} validation samples")

    logging.info(f"Training begun. Running for {num_epochs} epochs.")
    for epoch in tqdm(range(num_epochs), "Epoch"):
        model.train()
        for x, y in tqdm(train_loader, "Training", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_hat = model(x)
            loss: torch.Tensor = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.update(loss)
            train_metrics.update(y_hat, y)

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(valid_loader, "Validating", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_hat = model(x)
                loss = criterion(y_hat, y)

                valid_loss.update(loss)
                valid_metrics.update(y_hat, y)

        writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], epoch)

        writer.add_scalar("loss/train", train_loss.compute(), epoch)
        for metric_name, metric_value in train_metrics.compute().items():
            writer.add_scalar(metric_name, metric_value, epoch)

        valid_loss_epoch = valid_loss.compute()
        writer.add_scalar("loss/valid", valid_loss_epoch, epoch)
        for metric_name, metric_value in valid_metrics.compute().items():
            writer.add_scalar(metric_name, metric_value, epoch)

        if valid_loss_epoch < best_valid_loss:
            torch.save(model.state_dict(), f"{log_dir}/model_state_best_loss.pt")
            best_valid_loss = valid_loss_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == cfg.trainer.patience:
                logging.info("Maximum epochs without improvement reached. Early stopping.")
                break

        train_loss.reset()
        valid_loss.reset()
        train_metrics.reset()
        valid_metrics.reset()

        lr_scheduler.step()

    logging.info(f"Training ended. Best validation loss: {best_valid_loss:.4f}")
    torch.save(model.state_dict(), f"{log_dir}/model_state_last.pt")
    writer.close()


if __name__ == "__main__":
    main()
