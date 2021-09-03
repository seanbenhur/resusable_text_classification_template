import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_lrs(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# TODO : implement a different function w/o batch size
def get_accuracy(prediction, label, batch_size):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train_fn(
    train_loader,
    model,
    loss_fn,
    optimizer,
    epoch,
    scheduler,
    batch_size,
    grad_clip=True,
):
    model.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    # train_artifact = wandb.Artifact("dravidian-offensive",type="train logs")
    epoch_losses = []
    epoch_accuracy = []
    lrs = []
    for idx, inputs in progress_bar:
        ids = inputs["input_ids"].to(device, dtype=torch.long)
        mask = inputs["attention_mask"].to(device, dtype=torch.long)
        target = inputs["label"].to(device, dtype=torch.float)

        # Amp
        # with amp.autocast(enabled=True):
        optimizer.zero_grad()
        output = model(ids, mask)
        loss = loss_fn(output, target)
        loss.backward()

        acc = get_accuracy(output, target, batch_size)

        if grad_clip is not None:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        optimizer.step()

        # Record & update learning rate
        wandb.log(
            {
                "Learning Rate": get_lrs(optimizer),
                "Training Loss": loss,
                "Training accuracy": acc,
            }
        )
        lrs.append(get_lrs(optimizer))

        scheduler.step()

        progress_bar.set_description(
            f"Epoch: {epoch}, Train loss{loss}, Train accuracy{acc}"
        )

        epoch_losses.append(loss.item())
        epoch_accuracy.append(acc.item())

    return epoch_losses, epoch_accuracy, lrs


def evaluate_fn(val_loader, model, loss_fn, epoch, batch_size):
    model.eval()
    progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))

    epoch_losses = []
    epoch_accuracy = []
    with torch.no_grad():
        for idx, inputs in progress_bar:
            ids = inputs["input_ids"].to(device, dtype=torch.long)
            mask = inputs["attention_mask"].to(device, dtype=torch.long)
            target = inputs["label"].to(device, dtype=torch.float)

            output = model(ids, mask)
            loss = loss_fn(output, target)
            acc = get_accuracy(output, target, batch_size)

            wandb.log({"Valid Loss": loss, "Valid Accuracy": acc})
            progress_bar.set_description(
                f"Epoch: {epoch}, Valid loss{loss}, Valid accuracy{acc}"
            )

            epoch_losses.append(loss.item())
            epoch_accuracy.append(acc.item())

    return epoch_losses, epoch_accuracy
