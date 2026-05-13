"""
centralized.py — Modelo Centralizado (Baseline)

Punto 2.1 del TFG: entrenamiento centralizado sobre el dataset completo,
sin ningun mecanismo federado. Sirve como referencia para comparar contra
el sistema federado (2.2) y el sistema con agregacion segura (4.x).

Reutiliza integranamente utils.py: mismos modelos, mismo preprocesamiento,
mismas metricas. La unica diferencia es que aqui no hay rondas ni servidor.
Ambos modelos exportan exactamente el mismo esquema de CSV.
"""

import argparse
import csv
import time
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)

from utils import (
    load_data,
    Net, train_nn, test_nn,
    create_model, train_lr, test_lr,
)

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
NN_EPOCHS      = 20
METRICS_CSV_NN = "metrics_centralized_nn.csv"
METRICS_CSV_LR = "metrics_centralized_lr.csv"

# Esquema identico para ambos modelos
FIELDNAMES = [
    "epoch_or_run", "time", "loss",
    "accuracy", "precision", "recall", "f1", "mcc", "auc_roc",
    "tp", "tn", "fp", "fn",
    "precision_c0", "precision_c1",
    "recall_c0",    "recall_c1",
    "f1_c0",        "f1_c1",
    "model_bytes",
]

# --------------------------------------------------------------------------
# Argumentos
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Modelo Centralizado (Baseline)")
parser.add_argument(
    "--model", "-m",
    choices=["nn", "lr"],
    required=True,
    help="Modelo a usar: 'nn' (Red Neuronal) o 'lr' (Regresion Logistica)."
)
args = parser.parse_args()

# --------------------------------------------------------------------------
# Funcion auxiliar: metricas extendidas (confusion matrix + por clase)
# Se aplica a ambos modelos para garantizar el mismo esquema de CSV
# --------------------------------------------------------------------------
def _extended_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true,    y_pred, average=None, zero_division=0)
    f_per = f1_score(y_true,        y_pred, average=None, zero_division=0)
    return {
        "tp": float(tp), "tn": float(tn),
        "fp": float(fp), "fn": float(fn),
        "precision_c0": float(p_per[0]), "precision_c1": float(p_per[1]),
        "recall_c0":    float(r_per[0]), "recall_c1":    float(r_per[1]),
        "f1_c0":        float(f_per[0]), "f1_c1":        float(f_per[1]),
    }

# --------------------------------------------------------------------------
# 1. Cargar datos completos
# --------------------------------------------------------------------------
print("\n" + "="*60)
model_label = "Red Neuronal" if args.model == "nn" else "Regresion Logistica"
print(f"  MODELO CENTRALIZADO — {model_label}")
print("="*60)
print("Cargando dataset completo...")

X_train, X_test, y_train, y_test = load_data(use_dirichlet=False)
n_features = X_train.shape[1]
print(f"  Features:      {n_features}")
print(f"  Train samples: {X_train.shape[0]}")
print(f"  Test samples:  {X_test.shape[0]}")

# --------------------------------------------------------------------------
# 2. Instanciar modelo
# --------------------------------------------------------------------------
if args.model == "nn":
    model = Net(n_features)
    model_bytes = float(sum(p.numel() * p.element_size()
                            for p in model.parameters()))
else:
    model = create_model(n_features)
    model_bytes = float((n_features + 1) * 8)  # coef_ + intercept_ en float64

# --------------------------------------------------------------------------
# 3. Entrenamiento y evaluacion
# --------------------------------------------------------------------------
print(f"\n{'─'*60}")
history = []
t_start = time.time()

if args.model == "nn":
    print(f"Entrenando Red Neuronal ({NN_EPOCHS} epocas)...")

    for epoch in range(1, NN_EPOCHS + 1):
        t0 = time.time()
        train_nn(model, X_train, y_train, epochs=1)
        epoch_time = time.time() - t0

        metrics = test_nn(model, X_test, y_test)

        row = {
            "epoch_or_run": epoch,
            "time":         epoch_time,
            "loss":         metrics["loss"],
            "accuracy":     metrics["accuracy"],
            "precision":    metrics["precision"],
            "recall":       metrics["recall"],
            "f1":           metrics["f1"],
            "mcc":          metrics["mcc"],
            "auc_roc":      metrics["auc_roc"],
            "tp":           metrics["tp"],
            "tn":           metrics["tn"],
            "fp":           metrics["fp"],
            "fn":           metrics["fn"],
            "precision_c0": metrics["precision_c0"],
            "precision_c1": metrics["precision_c1"],
            "recall_c0":    metrics["recall_c0"],
            "recall_c1":    metrics["recall_c1"],
            "f1_c0":        metrics["f1_c0"],
            "f1_c1":        metrics["f1_c1"],
            "model_bytes":  metrics["model_bytes"],
        }
        history.append(row)

        print(f"  Epoca {epoch:02d}/{NN_EPOCHS} ({epoch_time:.2f}s) | "
              f"Acc: {metrics['accuracy']:.4f}  "
              f"F1: {metrics['f1']:.4f}  "
              f"MCC: {metrics['mcc']:.4f}  "
              f"AUC: {metrics['auc_roc']:.4f}  "
              f"Loss: {metrics['loss']:.6f}")

else:
    print("Entrenando Regresion Logistica...")
    t0 = time.time()
    train_lr(model, X_train, y_train)
    train_time = time.time() - t0

    metrics = test_lr(model, X_test, y_test)
    y_pred  = model.predict(X_test)
    ext     = _extended_metrics(y_test, y_pred)

    row = {
        "epoch_or_run": 1,
        "time":         train_time,
        "loss":         metrics["loss"],
        "accuracy":     metrics["accuracy"],
        "precision":    metrics["precision"],
        "recall":       metrics["recall"],
        "f1":           metrics["f1"],
        "mcc":          metrics["mcc"],
        "auc_roc":      metrics["auc_roc"],
        "model_bytes":  model_bytes,
        **ext,
    }
    history.append(row)
    print(f"  Completado en {train_time:.4f}s")

total_time = time.time() - t_start
final = history[-1]

# --------------------------------------------------------------------------
# 4. Resumen en consola
# --------------------------------------------------------------------------
print(f"\n{'─'*60}")
print("Evaluacion final sobre conjunto de test:")
print(f"  Accuracy:  {final['accuracy']:.4f}    Precision: {final['precision']:.4f}    Recall: {final['recall']:.4f}")
print(f"  F1-Score:  {final['f1']:.4f}    MCC:       {final['mcc']:.4f}    AUC-ROC: {final['auc_roc']:.4f}")
print(f"  Loss:      {final['loss']:.6f}")
print(f"  TP={final['tp']:.0f}  TN={final['tn']:.0f}  FP={final['fp']:.0f}  FN={final['fn']:.0f}")
print(f"  Clase 0 -> P={final['precision_c0']:.4f}  R={final['recall_c0']:.4f}  F1={final['f1_c0']:.4f}")
print(f"  Clase 1 -> P={final['precision_c1']:.4f}  R={final['recall_c1']:.4f}  F1={final['f1_c1']:.4f}")
print(f"\n  Tiempo total: {total_time:.2f}s")

# --------------------------------------------------------------------------
# 5. Exportar CSV
# --------------------------------------------------------------------------
csv_path = METRICS_CSV_NN if args.model == "nn" else METRICS_CSV_LR
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    w.writeheader()
    w.writerows(history)

print(f"\n{'='*60}")
print("  RESUMEN CENTRALIZADO")
print(f"{'='*60}")
print(f"  Modelo:         {model_label}")
print(f"  Features:       {n_features}")
print(f"  Train:          {X_train.shape[0]} muestras")
print(f"  Test:           {X_test.shape[0]} muestras")
if args.model == "nn":
    print(f"  Epocas:         {NN_EPOCHS}")
print(f"  Tiempo total:   {total_time:.2f}s")
print(f"{'─'*60}")
print(f"  Accuracy:       {final['accuracy']:.4f}")
print(f"  F1:             {final['f1']:.4f}")
print(f"  MCC:            {final['mcc']:.4f}")
print(f"  AUC-ROC:        {final['auc_roc']:.4f}")
print(f"  Loss:           {final['loss']:.6f}")
print(f"  Metricas:       {csv_path}")
print("="*60 + "\n")