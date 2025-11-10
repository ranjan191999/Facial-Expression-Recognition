import os, json, torch, random
import torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report
from stage1.models import resnet18_finetune
from stage1.data_utils import get_dataloaders

# ------------------- CONFIG -------------------
DATA_ROOT = "./data"
OUT_DIR = "./stage1/artifacts"
BATCH_SIZE = 32         # Larger batch → faster on 40GB GPU
EPOCHS = 6
LR = 2e-4
IMG_SIZE = 224
SEED = 42
# ----------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)

def train():
    train_dl, val_dl, classes = get_dataloaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)

    if len(train_dl) == 0 or len(val_dl) == 0:

        print(" Dataset empty or path incorrect.")

        print("Expected: ./data/train/<classes> and ./data/test/<classes>")
        return

    print(f" Found classes: {classes}")




    print(f" Training batches: {len(train_dl)} | Validation batches: {len(val_dl)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")





    model = resnet18_finetune(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()   
    best_acc = 0.0

    print("\n Starting Training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for i, (xb, yb) in enumerate(train_dl, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # ⚡ Mixed Precision Forward & Backward
            with autocast():
                loss = criterion(model(xb), yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)

            if i % 10 == 0:
                print(f"     Epoch {epoch} | Batch {i}/{len(train_dl)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_dl.dataset)

        # ---------- VALIDATION ----------
        model.eval()
        correct, total = 0, 0
        preds_all, labels_all = [], []

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                with autocast():
                    preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(yb.cpu().numpy())

        val_acc = correct / total if total > 0 else 0
        print(f"\n Epoch {epoch}/{EPOCHS} | TrainLoss: {train_loss:.4f} | ValAcc: {val_acc:.4f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            print(f" Model saved (ValAcc improved to {best_acc:.4f})")

    # ---------- SAVE REPORT ----------
    report = classification_report(labels_all, preds_all, target_names=classes, output_dict=True)
    with open(os.path.join(OUT_DIR, "val_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n Training Complete!")
    print(f" Best Validation Accuracy: {best_acc:.4f}")
    print(f" Model saved at: {os.path.join(OUT_DIR, 'best_model.pt')}")

if __name__ == "__main__":
    train()
