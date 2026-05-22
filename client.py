import flwr as fl
import torch
import time
import argparse
import warnings
from utils import (
    load_data,
    Net, train_nn, test_nn,
    create_model, model_to_params, params_to_model, train_lr, test_lr,
)
from secret_sharing import encode_parameters

warnings.filterwarnings("ignore")


# Argumentos
parser = argparse.ArgumentParser(description="Cliente de Aprendizaje Federado IoT")
parser.add_argument("--model", "-m", choices=["nn", "lr"], required=True)
parser.add_argument("--dirichlet", action="store_true", default=False)
parser.add_argument(
    "--secret-sharing",
    choices=["none", "shamir"],
    default="none",
    help="'none': FedAvg estándar. 'shamir': Secure Aggregation con Shamir (t,n).",
)
parser.add_argument(
    "--num-clients",
    type=int,
    default=None,
    help="Número total de clientes. Requerido con --secret-sharing shamir.",
)

parser.add_argument("--threshold", "-t", type=int, default=0, help="Umbral (t) para Shamir Secret Sharing")
parser.add_argument("--partition-id", type=int, default=0, help="ID de la partición de datos para este cliente (0-indexado).")
parser.add_argument("--num-partitions", type=int, default=2, help="Número total de particiones de datos (debe coincidir con el número de clientes si se usa --secret-sharing shamir).")

args = parser.parse_args()

# Validaciones


import math

n_shares     = None
ss_threshold = None
if args.secret_sharing == "shamir":
    n_shares = args.num_clients or args.num_partitions
    if n_shares is None:
        parser.error("--secret-sharing shamir requiere --num-clients.")
    ss_threshold = (args.threshold if args.threshold > 0
                    else math.ceil((n_shares + 1) / 2))
    if not (1 <= ss_threshold <= n_shares):
        parser.error(f"Umbral t={ss_threshold} inválido para n={n_shares}.")

# 1. Cargar datos
X_train, X_test, y_train, y_test = load_data(
    partition_id=args.partition_id,
    num_partitions=args.num_partitions,
    use_dirichlet=args.dirichlet,
)
n_features = X_train.shape[1]
label = f"Cliente {args.partition_id}" if args.dirichlet else "Cliente"
print(f"--> [{args.model.upper()}] {label} | Features: {n_features} | "
      f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# 2. Instanciar modelo
if args.model == "nn":
    model = Net(n_features)
else:
    model = create_model(n_features)

# Helper: parámetros del modelo en formato Flower (sin SS)
def _get_raw_parameters():
    if args.model == "nn":
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    else:
        return model_to_params(model)

# 3. Clase cliente Flower
class IoTClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # Devuelve parámetros normales (sin SS).
        # El servidor usa esto para inicializar el modelo global en la ronda 0.
        return _get_raw_parameters()

    def fit(self, parameters, config):
        # Cargar parámetros globales del servidor (siempre formato normal)
        if args.model == "nn":
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict  = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            t0 = time.time()
            train_nn(model, X_train, y_train, epochs=1)
        else:
            params_to_model(model, parameters)
            t0 = time.time()
            train_lr(model, X_train, y_train)

        train_time = time.time() - t0
        raw_params = _get_raw_parameters()

        if args.secret_sharing == "shamir":
            t_ss = time.time()
            encoded = encode_parameters(raw_params, n=n_shares, t=ss_threshold)
            ss_time = time.time() - t_ss
            return encoded, len(X_train), {
                "train_time":    train_time,
                "ss_split_time": ss_time,
            }

        return raw_params, len(X_train), {"train_time": train_time}

    def evaluate(self, parameters, config):
        # Los parámetros para evaluar son siempre normales (reconstruidos por el servidor).
        if args.model == "nn":
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict  = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            t0      = time.time()
            metrics = test_nn(model, X_test, y_test)
        else:
            params_to_model(model, parameters)
            t0      = time.time()
            metrics = test_lr(model, X_test, y_test)

        metrics["eval_time"] = time.time() - t0
        loss = metrics.pop("loss")
        return float(loss), len(X_test), {k: float(v) for k, v in metrics.items()}

# 4. Main
if __name__ == "__main__":
    try:
        fl.client.start_numpy_client(
            server_address="127.0.0.1:9090",
            client=IoTClient(),
        )
    except Exception as e:
        if "StatusCode.UNAVAILABLE" in str(e) or "Connection reset" in str(e):
            print(f"--> {label}: Servidor cerrado. Finalizando.")
        else:
            raise