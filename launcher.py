import subprocess
import sys
import time
import argparse
import os
import signal
from typing import List, Optional

# Configuracion

PYTHON_EXECUTABLE      = sys.executable
SERVER_SCRIPT          = "server.py"
CLIENT_SCRIPT          = "client.py"
DELAY_BETWEEN_CLIENTS  = 2   # Segundos entre lanzamiento de clientes
DELAY_SERVER_START     = 3   # Segundos de espera tras iniciar servidor

processes: List[subprocess.Popen] = []

# Manejo de senales

def signal_handler(signum, frame):
    print("\n\n[LAUNCHER] Interrupcion recibida. Deteniendo todos los procesos...")
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
    print("[LAUNCHER] Todos los procesos han sido detenidos.")

# Funciones de lanzamiento

def check_files_exist():
    missing = [f for f in [SERVER_SCRIPT, CLIENT_SCRIPT, "utils.py"]
               if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Archivos faltantes: {', '.join(missing)}")
        print("[ERROR] Asegurese de ejecutar desde el directorio correcto.")
        return False
    return True

def launch_server(num_clients: int, model_type: str) -> Optional[subprocess.Popen]:
    print(f"[LAUNCHER] Iniciando servidor [{model_type.upper()}]...")
    try:
        proc = subprocess.Popen([
            PYTHON_EXECUTABLE, SERVER_SCRIPT,
            "--model",   model_type,
            "--clients", str(num_clients),
        ])
        processes.append(proc)
        print(f"[LAUNCHER] Servidor iniciado (PID: {proc.pid}) | "
              f"Modelo: {model_type.upper()} | Clientes: {num_clients}")
        return proc
    except Exception as e:
        print(f"[ERROR] No se pudo iniciar el servidor: {e}")
        return None

def launch_client(client_id: int, num_clients: int,
                  model_type: str, use_dirichlet: bool) -> Optional[subprocess.Popen]:
    print(f"[LAUNCHER] Iniciando Cliente {client_id} [{model_type.upper()}]...")
    try:
        cmd = [PYTHON_EXECUTABLE, CLIENT_SCRIPT, "--model", model_type]

        if use_dirichlet:
            partition_id = client_id - 1   # 0-indexed
            cmd += [
                "--dirichlet",
                "--partition-id",   str(partition_id),
                "--num-partitions", str(num_clients),
            ]

        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"[LAUNCHER] Cliente {client_id} iniciado (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"[ERROR] No se pudo iniciar el cliente {client_id}: {e}")
        return None

# Orquestador principal

def launch_federated_system(num_clients: int, model_type: str,
                            use_dirichlet: bool):
    model_label = "Red Neuronal" if model_type == "nn" else "Regresion Logistica"
    particion   = f"Dirichlet (alpha=0.5)" if use_dirichlet else "Ninguna (dataset completo)"

    print("\n" + "="*60)
    print("LANZADOR DE SISTEMA DE APRENDIZAJE FEDERADO")
    print(f"Modelo:    {model_label}")
    print(f"Clientes:  {num_clients}")
    print(f"Particion: {particion}")
    print("="*60)

    if not check_files_exist():
        return

    # 1. Servidor
    server_proc = launch_server(num_clients, model_type)
    if server_proc is None:
        return

    print(f"[LAUNCHER] Esperando {DELAY_SERVER_START}s para que el servidor inicie...")
    time.sleep(DELAY_SERVER_START)

    if server_proc.poll() is not None:
        print("[ERROR] El servidor termino inesperadamente antes de lanzar clientes.")
        cleanup_processes()
        return

    # 2. Clientes
    for i in range(1, num_clients + 1):
        launch_client(i, num_clients, model_type, use_dirichlet)
        if i < num_clients:
            time.sleep(DELAY_BETWEEN_CLIENTS)

    print("\n" + "="*60)
    print("SISTEMA INICIADO - VIGILANDO PROCESOS")
    print("="*60)
    print("El launcher cerrara todo cuando el servidor termine.")

    # 3. Bucle de vigilancia
    try:
        while True:
            if server_proc.poll() is not None:
                print(f"\n[LAUNCHER] Servidor finalizado (Codigo: {server_proc.poll()}).")
                print("[LAUNCHER] Cerrando clientes restantes...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)
    finally:
        cleanup_processes()

# Main
def main():
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Launcher de Aprendizaje Federado")
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
        help="Numero de clientes a lanzar."
    )
    parser.add_argument(
        "--dirichlet",
        action="store_true",
        default=False,
        help=(
            "Si se especifica, cada cliente recibe una particion heterogenea "
            "del dataset mediante DirichletPartitioner (alpha=0.5). "
            "Si no se especifica, cada cliente carga el dataset completo."
        )
    )
    args = parser.parse_args()

    launch_federated_system(
        num_clients=args.clients,
        model_type=args.model,
        use_dirichlet=args.dirichlet,
    )

if __name__ == "__main__":
    main()