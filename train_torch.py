import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_training(history, save_path="training_curve.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Training curve saved to '{save_path}'")


def main():
    X = np.load("X_seq.npy")
    y_raw = np.load("y_seq.npy")
    print(f"[INFO] Loaded X={X.shape}, y={y_raw.shape}")
    print(f"[INFO] Unique classes: {np.unique(y_raw)}")

    # Diagnostic: check if gestures are separable
    for cls in np.unique(y_raw):
        mask = (y_raw == cls)
        mean_x = X[mask, :, 8].mean()  # x-coord of index tip (lm 8)
        print(f"[DIAG] {cls}: mean index.x = {mean_x:.3f}")

    # CRITICAL FIX: Use balanced sampling instead of just class weights
    le = LabelEncoder()
    le.fit(['erase', 'note'])  # Enforce 0=erase, 1=note
    y_enc = le.transform(y_raw)
    joblib.dump(le, "label_encoder.pkl")
    class_names = le.classes_.tolist()
    print(f"[INFO] Label mapping: {{0: '{class_names[0]}', 1: '{class_names[1]}'}}")

    # CRITICAL FIX: Use stratified sampling with balanced class distribution
    from sklearn.utils import resample
    
    # Separate classes
    X_erase = X[y_raw == "erase"]
    y_erase = y_raw[y_raw == "erase"]
    X_note = X[y_raw == "note"]
    y_note = y_raw[y_raw == "note"]
    
    # Balance the classes by downsampling majority class
    n_samples = min(len(X_erase), len(X_note))
    X_erase_balanced, y_erase_balanced = resample(X_erase, y_erase, n_samples=n_samples, random_state=42)
    X_note_balanced, y_note_balanced = resample(X_note, y_note, n_samples=n_samples, random_state=42)
    
    # Combine balanced data
    X_balanced = np.vstack([X_erase_balanced, X_note_balanced])
    y_balanced = np.hstack([y_erase_balanced, y_note_balanced])
    
    print(f"[INFO] Balanced dataset: {X_balanced.shape[0]} samples ({n_samples} per class)")

    X = np.transpose(X_balanced, (0, 2, 1))  # (N, 63, 16)
    y_raw = y_balanced

    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )
    y_train = torch.tensor(le.transform(y_train_raw), dtype=torch.long)
    y_val = torch.tensor(le.transform(y_val_raw), dtype=torch.long)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)

    # CRITICAL FIX: Remove class weights (use balanced data instead)
    criterion = nn.CrossEntropyLoss()  # No class weights needed with balanced data
    model = GestureNet(num_classes=len(class_names))
    
    # CRITICAL FIX: Lower learning rate and use SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    device = torch.device("cpu")
    model.to(device)
    X_train, X_val = X_train.to(device), X_val.to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)

    # Training
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15  # Increased patience

    print("\n" + "="*55)
    print(f"{'Epoch':<6} {'Train Loss':<10} {'Val Loss':<10} {'Val Acc':<8}")
    print("="*55)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        # CRITICAL FIX: Remove gradient clipping (was preventing learning)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_acc = (val_outputs.argmax(dim=1) == y_val).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:<6} {loss.item():<10.4f} {val_loss.item():<10.4f} {val_acc:<8.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    plot_training(history)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_enc = model(X_val).argmax(dim=1).cpu().numpy()
        pred = le.inverse_transform(pred_enc)
        y_true = y_val_raw

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, pred, target_names=class_names, zero_division=0))
    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(y_true, pred, labels=class_names)
    print(cm)
    print(f"\nLabels: {class_names}")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix: Gesture Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    print("[INFO] Confusion matrix visualization saved as 'confusion_matrix.png'")

    torch.save(model.state_dict(), "gesture_model.pth")
    print(f"\n[INFO] Best model (val acc: {best_val_acc:.2%}) saved.")


if __name__ == "__main__":
    from model import GestureNet
    main()