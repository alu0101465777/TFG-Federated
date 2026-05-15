import flwr as fl
import torch
import time
import argparse
import warnings
from utils import (
    load_data,
    # RNA
    Net, train_nn, test_nn,
    # RL
    create_model, model_to_params, params_to_model, train_lr, test_lr,
)

warnings.filterwarnings("ignore")

# Argumentos

parser = argparse.ArgumentParser(description="Cliente de Aprendizaje Federado IoT")
parser.add_argument(
    "--model", "-m",
    choices=["nn", "lr"],
    required=True,
    help="Modelo a usar: 'nn' (Red Neuronal) o 'lr' (Regresion Logistica)."
)
parser.add_argument(
    "--partition-id",
    type=int,
    default=None,
    help="ID de la particion de este cliente, 0-indexed. Solo necesario con --dirichlet."
)
parser.add_argument(
    "--num-partitions",
    type=int,
    default=None,
    help="Numero total de clientes. Solo necesario con --dirichlet."
)
parser.add_argument(
    "--dirichlet",
    action="store_true",
    default=False,
    help=(
        "Si se especifica, aplica particion Dirichlet (alpha=0.5). "
        "Cada cliente recibe una porcion heterogenea del dataset. "
        "Si no se especifica, cada cliente carga el dataset completo."
    )
)
args = parser.parse_args()

if args.dirichlet:
    if args.partition_id is None or args.num_partitions is None:
        parser.error("--dirichlet requiere --partition-id y --num-partitions.")

# 1. Cargar datos  (siempre arrays NumPy)
X_train, X_test, y_train, y_test = load_data(
    partition_id=args.partition_id,
    num_partitions=args.num_partitions,
    use_dirichlet=args.dirichlet,
)

n_features = X_train.shape[1]
label = f"Cliente {args.partition_id}" if args.dirichlet else "Cliente"
print(f"--> [{args.model.upper()}] {label} | Features: {n_features} | "
      f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 2. Instanciar modelo segun --model
if args.model == "nn":
    model = Net(n_features)
else:
    model = create_model(n_features)

# 3. Clase Cliente Flower
class IoTClient(fl.client.NumPyClient):
    
    def get_parameters(self, config):
        if args.model == "nn":
            return [val.cpu().numpy() for _, val in model.state_dict().items()]
        else:
            return model_to_params(model)

    def fit(self, parameters, config):
        # Cargar parametros globales en el modelo local
        if args.model == "nn":
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            t0 = time.time()
            train_nn(model, X_train, y_train, epochs=1)
        else:
            params_to_model(model, parameters)
            t0 = time.time()
            train_lr(model, X_train, y_train)

        train_time = time.time() - t0
        return self.get_parameters(config={}), len(X_train), {"train_time": train_time}

    def evaluate(self, parameters, config):
        # Cargar parametros globales y evaluar
        if args.model == "nn":
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            t0 = time.time()
            metrics = test_nn(model, X_test, y_test)
        else:
            params_to_model(model, parameters)
            t0 = time.time()
            metrics = test_lr(model, X_test, y_test)

        metrics["eval_time"] = time.time() - t0
        loss = metrics.pop("loss")
        return float(loss), len(X_test), {k: float(v) for k, v in metrics.items()}

# 4. Main
if __name__ == "__main__":
    try:
        fl.client.start_numpy_client(
            server_address="127.0.0.1:9090",
            client=IoTClient()
        )
    except Exception as e:
        if "StatusCode.UNAVAILABLE" in str(e) or "Connection reset" in str(e):
            print(f"--> {label}: Servidor cerrado. Finalizando.")
        else:
            raise