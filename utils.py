import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, log_loss, precision_score, recall_score,
                             f1_score, confusion_matrix, matthews_corrcoef, roc_auc_score)

# Parametro de particion Dirichlet

# Alpha controla el grado de heterogeneidad entre clientes: ( Importante )
#   alpha pequeño (0.1) -> muy no-IID (cada cliente ve casi una sola clase)
#   alpha = 0.5         -> heterogeneidad moderada ( Se elige esta opcion porque es un buen punto intermedio)
#   alpha grande (1.0+) -> casi IID
DIRICHLET_ALPHA = 0.5

# BLOQUE 1: CARGA DE DATOS (compartido por ambos modelos)
# Siempre devuelve arrays NumPy: X_train, X_test, y_train, y_test

def _build_preprocessors():
    df0 = pd.read_csv("clase0.csv", nrows=5000)
    df1 = pd.read_csv("clase1.csv", nrows=5000)
    df0["Target"] = 0
    df1["Target"] = 1
    df = pd.concat([df0, df1], ignore_index=True)

    df.columns = df.columns.str.strip()
    for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "SimillarHTTP"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X_full = df.drop(columns=["Target"]).values.astype(np.float32)

    selector = VarianceThreshold(threshold=0)
    try:
        selector.fit(X_full)
    except ValueError:
        selector = None

    X_filtered = selector.transform(X_full) if selector else X_full
    scaler = StandardScaler()
    scaler.fit(X_filtered)

    return selector, scaler


def _load_full_data():
    # Carga el dataset completo. Cada cliente ve todos los datos (escenario IID).
    df0 = pd.read_csv("clase0.csv", nrows=5000)
    df1 = pd.read_csv("clase1.csv", nrows=5000)
    df0["Target"] = 0
    df1["Target"] = 1
    df = pd.concat([df0, df1], ignore_index=True)

    df.columns = df.columns.str.strip()
    for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "SimillarHTTP"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df.drop(columns=["Target"]).values, df["Target"].values


def _load_dirichlet_partition(partition_id: int, num_partitions: int):
    
   # Carga la particion Dirichlet de este cliente.

    from datasets import Dataset as HFDataset
    from flwr_datasets.partitioner import DirichletPartitioner

    df0 = pd.read_csv("clase0.csv", nrows=5000)
    df1 = pd.read_csv("clase1.csv", nrows=5000)
    df0["Target"] = 0
    df1["Target"] = 1
    df = pd.concat([df0, df1], ignore_index=True)

    df.columns = df.columns.str.strip()
    for col in ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "SimillarHTTP"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    hf_dataset = HFDataset.from_pandas(df, preserve_index=False)

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="Target",
        alpha=DIRICHLET_ALPHA,
        min_partition_size=10,
        self_balancing=True,
    )
    partitioner.dataset = hf_dataset
    partition_df = partitioner.load_partition(partition_id).to_pandas()

    class_counts = partition_df["Target"].value_counts().to_dict()
    print(
        f"--> [Cliente {partition_id}] Dirichlet (alpha={DIRICHLET_ALPHA}): "
        f"{len(partition_df)} muestras | "
        f"Clase 0 (normal): {class_counts.get(0, 0)} | "
        f"Clase 1 (ataque): {class_counts.get(1, 0)}"
    )

    return partition_df.drop(columns=["Target"]).values, partition_df["Target"].values


def load_data(partition_id: int = None, num_partitions: int = None,
              use_dirichlet: bool = False):

    # Punto de entrada unico para la carga de datos. Usado por ambos modelos.

    # Devuelve siempre arrays NumPy: X_train, X_test, y_train, y_test.
    # - La RNA los envuelve en DataLoaders internamente en sus funciones train/test.
    # - La RL los usa directamente.


    if use_dirichlet:
        X, y = _load_dirichlet_partition(partition_id, num_partitions)
    else:
        X, y = _load_full_data()

    selector, scaler = _build_preprocessors()
    if selector is not None:
        X = selector.transform(X)
    X = scaler.transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# BLOQUE 2: RED NEURONAL (PyTorch)

class IoTDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_size_bytes(net):
    return sum(p.numel() * p.element_size() for p in net.parameters())


def train_nn(net, X_train, y_train, epochs=1):
    loader = DataLoader(IoTDataset(X_train, y_train), batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for data, labels in loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()


def test_nn(net, X_test, y_test):
    loader = DataLoader(IoTDataset(X_test, y_test), batch_size=32)
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    net.eval()
    with torch.no_grad():
        for data, labels in loader:
            outputs = net(data)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    n = len(all_labels)
    if n == 0:
        return {"accuracy": 0.0, "loss": 0.0}

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "accuracy":     float((y_pred == y_true).sum() / n),
        "loss":         total_loss / n,
        "precision":    float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1":           float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_true, y_pred)),
        "auc_roc":      float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
        "precision_c0": float(p_per[0]), "precision_c1": float(p_per[1]),
        "recall_c0":    float(r_per[0]), "recall_c1":    float(r_per[1]),
        "f1_c0":        float(f_per[0]), "f1_c1":        float(f_per[1]),
        "model_bytes":  float(get_model_size_bytes(net)),
    }


# BLOQUE 3: REGRESION LOGISTICA (scikit-learn)

def create_model(n_features: int) -> LogisticRegression:
    model = LogisticRegression(max_iter=200, solver="lbfgs", warm_start=True)
    model.classes_ = np.array([0, 1])
    model.coef_ = np.zeros((1, n_features), dtype=np.float64)
    model.intercept_ = np.zeros(1, dtype=np.float64)
    return model


def model_to_params(model: LogisticRegression):
    return [model.coef_.copy(), model.intercept_.copy()]


def params_to_model(model: LogisticRegression, params):
    model.coef_ = params[0].copy()
    model.intercept_ = params[1].copy()
    return model


def train_lr(model: LogisticRegression, X_train, y_train) -> LogisticRegression:
    # Comprobar si el cliente tiene al menos 2 clases
    clases_presentes = np.unique(y_train)
    if len(clases_presentes) < 2:
        print(f"  [AVISO] Entrenamiento omitido: el cliente solo tiene datos de la clase {clases_presentes[0]}.")
        return model
        
    model.fit(X_train, y_train)
    return model


def test_lr(model: LogisticRegression, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    return {
        "accuracy":  float(accuracy_score(y_test, preds)),
        "loss":      float(log_loss(y_test, proba, labels=[0, 1])),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_test, preds, average="weighted", zero_division=0)),
        "f1":        float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_test, preds)),
        "auc_roc":   float(roc_auc_score(y_test, proba[:, 1]))
                     if len(np.unique(y_test)) > 1 else 0.5,
    }