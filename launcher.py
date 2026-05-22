#!/usr/bin/env python3

import subprocess
import sys
import time
import argparse
import os
import signal
import math
from typing import List, Optional

PYTHON_EXECUTABLE     = sys.executable
SERVER_SCRIPT         = "server.py"
CLIENT_SCRIPT         = "client.py"
DELAY_BETWEEN_CLIENTS = 2
DELAY_SERVER_START    = 3

processes: List[subprocess.Popen] = []

def signal_handler(signum, frame):
    print("\n\n[LAUNCHER] Interrupción recibida. Deteniendo procesos...")
    cleanup_processes()
    sys.exit(0)

def cleanup_processes():
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception as e:
                print(f"[LAUNCHER] Error al terminar proceso: {e}")
    print("[LAUNCHER] Todos los procesos detenidos.")

def check_files_exist():
    required = [SERVER_SCRIPT, CLIENT_SCRIPT, "utils.py", "secret_sharing.py"]
    missing  = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Archivos faltantes: {', '.join(missing)}")
        return False
    return True

def launch_server(num_clients: int, model_type: str,
                  ss_mode: str, ss_threshold: int) -> Optional[subprocess.Popen]:
    print(f"[LAUNCHER] Iniciando servidor [{model_type.upper()}]"
          + (" [SecAgg Shamir]" if ss_mode == "shamir" else "") + "...")
    cmd = [
        PYTHON_EXECUTABLE, SERVER_SCRIPT,
        "--model",          model_type,
        "--clients",        str(num_clients),
        "--secret-sharing", ss_mode,
    ]
    if ss_mode == "shamir" and ss_threshold > 0:
        cmd += ["--threshold", str(ss_threshold)]
    try:
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"[LAUNCHER] Servidor iniciado (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"[ERROR] No se pudo iniciar el servidor: {e}")
        return None

def launch_client(client_id: int, num_clients: int, model_type: str,
                  use_dirichlet: bool, ss_mode: str,
                  ss_threshold: int) -> Optional[subprocess.Popen]:
    print(f"[LAUNCHER] Iniciando Cliente {client_id} [{model_type.upper()}]...")
    cmd = [PYTHON_EXECUTABLE, CLIENT_SCRIPT, "--model", model_type]
    if use_dirichlet:
        cmd += ["--dirichlet",
                "--partition-id",   str(client_id - 1),
                "--num-partitions", str(num_clients)]
    if ss_mode == "shamir":
        cmd += ["--secret-sharing", "shamir",
                "--num-clients",    str(num_clients)]
        if ss_threshold > 0:
            cmd += ["--threshold", str(ss_threshold)]
    try:
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"[LAUNCHER] Cliente {client_id} iniciado (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"[ERROR] No se pudo iniciar el cliente {client_id}: {e}")
        return None

def launch_federated_system(num_clients: int, model_type: str,
                             use_dirichlet: bool, ss_mode: str,
                             ss_threshold: int):
    model_label = "Red Neuronal" if model_type == "nn" else "Regresion Logistica"
    particion   = "Dirichlet (alpha=0.5)" if use_dirichlet else "Ninguna (dataset completo)"
    if ss_mode == "shamir":
        t_eff    = ss_threshold if ss_threshold > 0 else math.ceil((num_clients + 1) / 2)
        ss_label = f"Shamir (t={t_eff}, n={num_clients})"
    else:
        ss_label = "Ninguno (FedAvg estándar)"

    print("\n" + "="*60)
    print("LANZADOR DE SISTEMA DE APRENDIZAJE FEDERADO")
    print(f"Modelo:     {model_label}")
    print(f"Clientes:   {num_clients}")
    print(f"Partición:  {particion}")
    print(f"SecAgg:     {ss_label}")
    print("="*60)

    if not check_files_exist():
        return

    server_proc = launch_server(num_clients, model_type, ss_mode, ss_threshold)
    if server_proc is None:
        return

    print(f"[LAUNCHER] Esperando {DELAY_SERVER_START}s...")
    time.sleep(DELAY_SERVER_START)

    if server_proc.poll() is not None:
        print("[ERROR] El servidor terminó inesperadamente.")
        cleanup_processes()
        return

    for i in range(1, num_clients + 1):
        launch_client(i, num_clients, model_type, use_dirichlet, ss_mode, ss_threshold)
        if i < num_clients:
            time.sleep(DELAY_BETWEEN_CLIENTS)

    print("\n[LAUNCHER] Sistema iniciado. Esperando al servidor...")

    try:
        while True:
            if server_proc.poll() is not None:
                print(f"\n[LAUNCHER] Servidor finalizado.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
    finally:
        cleanup_processes()

def main():
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Launcher de Aprendizaje Federado")
    parser.add_argument("--model", "-m", choices=["nn", "lr"], required=True)
    parser.add_argument("--clients", "-c", type=int, default=2)
    parser.add_argument("--dirichlet", action="store_true", default=False)
    parser.add_argument("--threshold", "-t", type=int, default=0, help="Umbral (t) para Shamir Secret Sharing")
    parser.add_argument(
        "--secret-sharing",
        choices=["none", "shamir"],
        default="none",
        help="'none': FedAvg estándar. 'shamir': Secure Aggregation con Shamir.",
    )
    args = parser.parse_args()
    launch_federated_system(
        num_clients=args.clients,
        model_type=args.model,
        use_dirichlet=args.dirichlet,
        ss_mode=args.secret_sharing,
        ss_threshold=args.threshold,
    )

if __name__ == "__main__":
    main()