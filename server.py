import flwr as fl
import os
import csv
import time
import numpy as np
from typing import List, Optional


# Configuracion

PATIENCE        = 3
MIN_IMPROVEMENT = 0.001
MAX_ROUNDS      = 50

# Metricas rastreadas por modelo
TRACKED_METRICS_NN = [
    "accuracy", "precision", "recall", "f1", "mcc", "auc_roc",
    "tp", "tn", "fp", "fn",
    "precision_c0", "precision_c1", "recall_c0", "recall_c1", "f1_c0", "f1_c1",
    "model_bytes", "eval_time",
]
TRACKED_METRICS_LR = [
    "accuracy", "precision", "recall", "f1", "mcc", "auc_roc",
]


# 1. Registro de Metricas

class TrainingMetrics:
    def __init__(self, tracked_metrics: List[str], metrics_csv: str, model_type: str):
        self.tracked_metrics  = tracked_metrics
        self.metrics_csv      = metrics_csv
        self.model_type       = model_type
        self.rounds: List[dict] = []
        self.start_time: Optional[float] = None
        self.round_start_time: Optional[float] = None
        self._init_csv()

    def _init_csv(self):
        extra = (["client_train_time_mean", "client_train_time_std",
                  "comm_bytes_total", "accuracy_variance"]
                 if self.model_type == "nn"
                 else ["client_train_time_mean", "accuracy_variance"])
        header = ["round", "round_time", "loss"] + self.tracked_metrics + extra
        with open(self.metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def start_training(self):
        self.start_time = time.time()
        model_label = "Red Neuronal" if self.model_type == "nn" else "Regresion Logistica"
        print("\n" + "="*70)
        print(f"  INICIO DEL ENTRENAMIENTO FEDERADO ({model_label})")
        print("="*70)

    def start_round(self, round_num: int):
        self.round_start_time = time.time()

    def end_round(self, round_num: int, loss: float, agg: dict,
                  client_accs: List[float], client_train_times: List[float],
                  comm_bytes: int = 0):
        round_time = time.time() - self.round_start_time if self.round_start_time else 0.0
        acc_var    = float(np.var(client_accs))    if len(client_accs) > 1    else 0.0
        train_mean = float(np.mean(client_train_times)) if client_train_times else 0.0
        train_std  = float(np.std(client_train_times))  if client_train_times else 0.0

        record = {
            "round": round_num, "round_time": round_time, "loss": loss,
            "client_train_time_mean": train_mean, "client_train_time_std": train_std,
            "comm_bytes_total": comm_bytes, "accuracy_variance": acc_var,
            **{k: agg.get(k, 0.0) for k in self.tracked_metrics},
        }
        self.rounds.append(record)

        # Escribir fila CSV
        if self.model_type == "nn":
            extra_cols = ["client_train_time_mean", "client_train_time_std",
                          "comm_bytes_total", "accuracy_variance"]
        else:
            extra_cols = ["client_train_time_mean", "accuracy_variance"]

        with open(self.metrics_csv, "a", newline="") as f:
            cols = ["round", "round_time", "loss"] + self.tracked_metrics + extra_cols
            csv.writer(f).writerow([record.get(c, 0.0) for c in cols])

        # Imprimir resumen de ronda
        print(f"\n{'─'*70}")
        print(f"  RONDA {round_num} COMPLETADA  ({round_time:.2f}s)")
        print(f"{'─'*70}")
        print(f"  Accuracy:  {agg.get('accuracy',0):.4f}    "
              f"Precision: {agg.get('precision',0):.4f}    "
              f"Recall: {agg.get('recall',0):.4f}")
        print(f"  F1-Score:  {agg.get('f1',0):.4f}    "
              f"MCC:       {agg.get('mcc',0):.4f}    "
              f"AUC-ROC: {agg.get('auc_roc',0):.4f}")
        print(f"  Loss:      {loss:.6f}")

        if self.model_type == "nn":
            print(f"  Model:     {agg.get('model_bytes',0)/1024:.1f} KB")
            tp = agg.get('tp', 0); tn = agg.get('tn', 0)
            fp = agg.get('fp', 0); fn = agg.get('fn', 0)
            print(f"  Confusion: TP={tp:.0f}  TN={tn:.0f}  FP={fp:.0f}  FN={fn:.0f}")
            print(f"  Clase 0 -> P={agg.get('precision_c0',0):.4f}  "
                  f"R={agg.get('recall_c0',0):.4f}  F1={agg.get('f1_c0',0):.4f}")
            print(f"  Clase 1 -> P={agg.get('precision_c1',0):.4f}  "
                  f"R={agg.get('recall_c1',0):.4f}  F1={agg.get('f1_c1',0):.4f}")
            print(f"  Comm overhead: {comm_bytes/1024:.1f} KB | "
                  f"Train time (media): {train_mean:.4f}s")
        else:
            print(f"  Train time (media clientes): {train_mean:.4f}s")

        if len(client_accs) > 1:
            print(f"  Varianza accuracy entre clientes: {acc_var:.6f}")

    def print_summary(self):
        total_time = time.time() - self.start_time if self.start_time else 0.0
        accs   = [r["accuracy"] for r in self.rounds]
        losses = [r["loss"]     for r in self.rounds]
        best_r = max(self.rounds, key=lambda r: r["accuracy"])

        model_label = ("Red Neuronal (64-32-2)" if self.model_type == "nn"
                       else "Regresion Logistica (warm_start)")

        print("\n" + "="*70)
        print("  RESUMEN FINAL DEL ENTRENAMIENTO")
        print("="*70)
        print(f"  Modelo:              {model_label}")
        print(f"  Rondas completadas:  {len(self.rounds)}")
        print(f"  Tiempo total:        {total_time:.2f} s")
        print(f"{'─'*70}")
        print(f"  RENDIMIENTO")
        print(f"  Mejor Accuracy:      {best_r['accuracy']:.4f} (Ronda {best_r['round']})")
        print(f"  Mejor F1:            {best_r['f1']:.4f}")
        print(f"  Mejor MCC:           {best_r['mcc']:.4f}")
        print(f"  Mejor AUC-ROC:       {best_r['auc_roc']:.4f}")
        print(f"  Loss final:          {losses[-1]:.6f}")
        print(f"{'─'*70}")
        print(f"  COSTE COMPUTACIONAL")
        avg_round = np.mean([r["round_time"]             for r in self.rounds])
        avg_train = np.mean([r["client_train_time_mean"] for r in self.rounds])
        print(f"  Tiempo medio/ronda:  {avg_round:.2f} s")
        print(f"  Tiempo medio train:  {avg_train:.4f} s")
        if self.model_type == "nn":
            total_comm = sum(r["comm_bytes_total"] for r in self.rounds)
            print(f"  Comunicacion total:  {total_comm/1024:.1f} KB")
            print(f"  Tamano modelo:       {best_r['model_bytes']/1024:.1f} KB")
        print(f"{'─'*70}")
        print(f"  ESTABILIDAD FEDERADA")
        acc_vars = [r["accuracy_variance"] for r in self.rounds]
        print(f"  Varianza media acc:  {np.mean(acc_vars):.6f}")
        print(f"  Desv. accuracy:      {np.std(accs):.6f}")
        print(f"{'─'*70}")
        print(f"  Metricas exportadas: {self.metrics_csv}")
        print("="*70 + "\n")

    def should_stop(self) -> bool:
        accs = [r["accuracy"] for r in self.rounds]
        if len(accs) < PATIENCE + 1:
            return False
        improvement = max(accs[-PATIENCE:]) - max(accs[:-PATIENCE])
        if improvement < MIN_IMPROVEMENT:
            print(f"\n  [EARLY STOPPING] Mejora: {improvement:.6f} < {MIN_IMPROVEMENT}")
            return True
        return False


# 2. Estrategia con Early Stopping

class EarlyStoppingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, training_metrics: TrainingMetrics,
                 tracked_metrics: List[str], model_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_metrics       = training_metrics
        self.tracked_metrics        = tracked_metrics
        self.model_type             = model_type
        self.should_continue        = True
        self.last_fit_bytes         = 0
        self.last_client_train_times: List[float] = []

    def configure_fit(self, server_round, parameters, client_manager):
        if not self.should_continue:
            return None
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        if not self.should_continue:
            return None
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        if server_round == 1:
            self.training_metrics.start_training()
        self.training_metrics.start_round(server_round)

        self.last_fit_bytes = 0
        self.last_client_train_times = []
        for _, fit_res in results:
            if self.model_type == "nn":
                for t in fit_res.parameters.tensors:
                    self.last_fit_bytes += len(t)
            self.last_client_train_times.append(
                fit_res.metrics.get("train_time", 0.0)
            )

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        total = sum(r.num_examples for _, r in results)

        agg = {}
        for key in self.tracked_metrics:
            vals = [(r.num_examples, r.metrics.get(key, 0.0)) for _, r in results]
            agg[key] = sum(n * v for n, v in vals) / total if total else 0.0

        agg_loss = (sum(r.num_examples * r.loss for _, r in results) / total
                    if total else 0.0)
        client_accs = [r.metrics.get("accuracy", 0.0) for _, r in results]

        self.training_metrics.end_round(
            round_num=server_round,
            loss=agg_loss,
            agg=agg,
            client_accs=client_accs,
            client_train_times=self.last_client_train_times,
            comm_bytes=self.last_fit_bytes,
        )

        stop_conv  = self.training_metrics.should_stop()
        stop_limit = (server_round >= MAX_ROUNDS)

        if stop_conv or stop_limit:
            self.should_continue = False
            reason = "CONVERGENCIA" if stop_conv else f"LIMITE DE RONDAS ({MAX_ROUNDS})"
            print(f"\n  [INFO] Fin por {reason}.")
            self.training_metrics.print_summary()
            print("  [SERVER] Cerrando proceso de servidor...")
            os._exit(0)

        return agg_loss, {"accuracy": agg["accuracy"]}


# Main

def run_server(min_clients: int, model_type: str):
    if model_type == "nn":
        tracked_metrics = TRACKED_METRICS_NN
        metrics_csv     = "metrics_neural_network.csv"
    else:
        tracked_metrics = TRACKED_METRICS_LR
        metrics_csv     = "metrics_logistic_regression.csv"

    training_metrics = TrainingMetrics(tracked_metrics, metrics_csv, model_type)

    def weighted_average(metrics):
        examples = [n for n, _ in metrics]
        if not examples:
            return {"accuracy": 0}
        total = sum(examples)
        return {k: sum(n * m.get(k, 0.0) for n, m in metrics) / total
                for k in tracked_metrics}

    strategy = EarlyStoppingStrategy(
        training_metrics=training_metrics,
        tracked_metrics=tracked_metrics,
        model_type=model_type,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    model_label = "Red Neuronal" if model_type == "nn" else "Regresion Logistica"
    print(f"Iniciando servidor ({model_label}). "
          f"Rondas Max: {MAX_ROUNDS}. Paciencia: {PATIENCE}")
    try:
        fl.server.start_server(
            server_address="0.0.0.0:9090",
            config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Excepcion en servidor: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Servidor de Aprendizaje Federado")
    parser.add_argument(
        "--model", "-m",
        choices=["nn", "lr"],
        required=True,
        help="Modelo a usar: 'nn' (Red Neuronal) o 'lr' (Regresion Logistica)."
    )
    parser.add_argument(
        "--clients", "-c",
        type=int,
        default=2,
        help="Numero de clientes requeridos para iniciar el entrenamiento."
    )
    args = parser.parse_args()
    run_server(min_clients=args.clients, model_type=args.model)