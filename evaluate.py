import os, torch
from torch.cuda.amp import autocast
from sklearn.metrics import classification_report
from stage1.models import resnet18_finetune
from stage1.data_utils import get_dataloaders

# ------------------- CONFIG -------------------
DATA_ROOT = "./data"
OUT_DIR = "./stage1/artifacts"
BATCH_SIZE = 32
IMG_SIZE = 224
# ----------------------------------------------

def evaluate():
    _, val_dl, classes = get_dataloaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)
    if len(val_dl) == 0:
        print("❌ Validation data is empty! Check ./data/train and ./data/test folders.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    model = resnet18_finetune(num_classes=len(classes))
    model_path = os.path.join(OUT_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Run train.py first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("\n🚀 Evaluating the trained model...\n")

    preds_all, labels_all = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            with autocast():
                preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())

    acc = correct / total if total > 0 else 0
    print(f"✅ Evaluation Complete! Overall Accuracy: {acc:.4f}")

    report = classification_report(labels_all, preds_all, target_names=classes)
    print("\n📊 Classification Report:\n")
    print(report)

    with open(os.path.join(OUT_DIR, "eval_report.txt"), "w") as f:
        f.write(report)
    print(f"💾 Report saved at: {os.path.join(OUT_DIR, 'eval_report.txt')}")

if __name__ == "__main__":
    evaluate()