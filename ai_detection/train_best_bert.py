import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, classification_report,
    roc_auc_score, brier_score_loss, fbeta_score
)

sys.path.append(os.path.abspath(os.path.join("..")))
from models import pretrained_transformer


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


def c_at_1(y_true, y_prob, threshold=0.5, uncertain_band=0.05):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = np.full_like(y_true, -1)
    uncertain = np.abs(y_prob - threshold) <= uncertain_band
    y_pred[~uncertain] = (y_prob[~uncertain] >= threshold).astype(int)
    n = len(y_true)
    n_correct = np.sum(y_pred == y_true)
    n_unanswered = np.sum(uncertain)
    return (1 / n) * (n_correct + n_unanswered * (n_correct / n))


def composite_metric(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob, dtype=float)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    roc   = roc_auc_score(y_true, y_prob)
    brier = 1 - brier_score_loss(y_true, y_prob)
    f1    = f1_score(y_true, y_pred)
    f05u  = fbeta_score(y_true, y_pred, beta=0.5)
    c1    = c_at_1(y_true, y_prob)
    return np.mean([roc, brier, f1, f05u, c1])


def full_report(y_true, y_prob, label=""):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob, dtype=float)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    comp  = composite_metric(y_true, y_prob)
    f1mac = f1_score(y_true, y_pred, average="macro")
    roc   = roc_auc_score(y_true, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Composite metric : {comp:.4f}")
    print(f"  F1-macro         : {f1mac:.4f}")
    print(f"  ROC-AUC          : {roc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["humano", "IA"]))
    return comp, f1mac, roc


df = pd.read_csv("data/dataset_truncated.csv")
X = df["texto"]
y = (df["clase"] == "susp").astype(int)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15,         
    stratify=y_train_full,
    random_state=SEED
)

print(f"Train : {len(X_train):>5} examples")
print(f"Val   : {len(X_val):>5} examples")
print(f"Test  : {len(X_test):>5} examples")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_dataset = pretrained_transformer.BERTDataset(X_train, y_train, tokenizer)
val_dataset   = pretrained_transformer.BERTDataset(X_val,   y_val,   tokenizer)
test_dataset  = pretrained_transformer.BERTDataset(X_test,  y_test,  tokenizer)

BATCH_SIZE = 16  

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE) 
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)


def train_bert(
    model,
    train_loader,
    val_loader,
    epochs=8,            
    lr=2e-5,
    patience=3,
    save_path=None,
    label="BERT"
):
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_score = -np.inf
    patience_counter = 0
    history = {"train_loss": [], "val_composite": [], "val_f1mac": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                logits = model(input_ids, attention_mask)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                lbls   = batch["labels"].numpy()
                all_probs.extend(probs)
                all_labels.extend(lbls)

        val_comp = composite_metric(all_labels, all_probs)
        val_f1   = f1_score(all_labels, (np.array(all_probs) >= 0.5).astype(int), average="macro")
        history["train_loss"].append(avg_loss)
        history["val_composite"].append(val_comp)
        history["val_f1mac"].append(val_f1)

        print(
            f"[{label}] Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  val_composite={val_comp:.4f}  val_f1mac={val_f1:.4f}"
        )

        if val_comp > best_val_score:
            best_val_score = val_comp
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✔ Best model saved → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹ Early stopping in epoch {epoch}")
                break

    training_time = time.time() - t0
    print(f"\n[{label}] Training completed {training_time:.1f}s")
    return best_val_score, history, training_time



print("\n" + "="*60)
print("  ABLATION: CLS Pooling vs Mean Pooling")
print("="*60)

ablation_results = {}
os.makedirs("../results", exist_ok=True)

for pooling in ["cls", "mean"]:
    print(f"\n▶ Training BERT with pooling='{pooling}'...")
    model_ab = pretrained_transformer.TransformerClassifier(
        model_name="bert-base-uncased",
        output_dim=2,
        pooling=pooling,
        freeze_encoder=False
    )
    score, _, _ = train_bert(
        model_ab,
        train_loader,
        val_loader,
        epochs=8,
        lr=2e-5,
        patience=3,
        save_path=f"../results/bert_{pooling}_ablation.pt",
        label=f"BERT-{pooling.upper()}"
    )
    ablation_results[pooling] = score
    print(f"  → Val composite ({pooling}): {score:.4f}")

best_pooling = max(ablation_results, key=ablation_results.get)
print(f"\n Best pooling: '{best_pooling}' "
      f"(score={ablation_results[best_pooling]:.4f})")


BEST_MODEL_PATH = "../results/bert_best_model.pt"

print("\n" + "="*60)
print(f"  FINAL TRAINING  (pooling='{best_pooling}', freeze=False)")
print("="*60)

model_best = pretrained_transformer.TransformerClassifier(
    model_name="bert-base-uncased",
    output_dim=2,
    pooling=best_pooling,
    freeze_encoder=False
)

best_val_score, history, training_time = train_bert(
    model_best,
    train_loader,
    val_loader,
    epochs=8,
    lr=2e-5,
    patience=3,
    save_path=BEST_MODEL_PATH,
    label="BERT-FINAL"
)


print("\n" + "="*60)
print("  TEST EVAL (with best checkpoint)")
print("="*60)

model_best.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model_best.to(DEVICE)
model_best.eval()

all_probs, all_labels = [], []
t_inf = time.time()
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model_best(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch["labels"].numpy())

inference_time = time.time() - t_inf

comp_test, f1mac_test, roc_test = full_report(
    all_labels, all_probs,
    label=f"TEST — BERT ({best_pooling.upper()} pooling, fine-tuned)"
)


summary = {
    "best_pooling":    best_pooling,
    "val_composite":   round(best_val_score, 4),
    "test_composite":  round(comp_test, 4),
    "test_f1_macro":   round(f1mac_test, 4),
    "test_roc_auc":    round(roc_test, 4),
    "training_time_s": round(training_time, 1),
    "inference_time_s":round(inference_time, 3),
    "model_path":      BEST_MODEL_PATH
}

print("\n" + "="*60)
print(" FINAL SUMMARY")
print("="*60)
for k, v in summary.items():
    print(f"  {k:<22}: {v}")

pd.DataFrame([summary]).to_csv("../results/bert_best_summary.csv", index=False)
print(f"\nSaved model in : {BEST_MODEL_PATH}")
print("Saved summary in: ../results/bert_best_summary.csv")