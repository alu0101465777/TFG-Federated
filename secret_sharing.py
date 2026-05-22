import numpy as np
from typing import List, Tuple

PRIME    = (1 << 31) - 1  
PRIME_NP = np.int64(PRIME)
SCALE    = 10_000

# Genera un array de tamaño d con valores aleatorios en Z_P
def _uniform_zp(d: int) -> np.ndarray:
    return np.random.randint(0, PRIME, size=d, dtype=np.int64)

# Calcula el inverso multiplicativo de a en Z_P usando inverso modular.
def _modinv(a: int) -> int:
    return pow(int(a) % PRIME, PRIME - 2, PRIME)

# Quantiza un array de pesos a enteros en Z_P, escalando por SCALE para conservar la  precisión de decimales
def _quantize(weights: np.ndarray) -> np.ndarray:
    flat = np.round(weights.flatten().astype(np.float64) * SCALE).astype(np.int64) 
    return flat % PRIME_NP

# Dequantiza un array de enteros en Z_P a pesos flotantes, revirtiendo el escalado.
def _dequantize(arr: np.ndarray, shape: tuple) -> np.ndarray:
    arr    = arr.astype(np.int64) % PRIME_NP
    half_p = np.int64(PRIME // 2)
    signed = np.where(arr > half_p, arr - PRIME_NP, arr)
    return (signed.astype(np.float64) / SCALE).astype(np.float32).reshape(shape)

# Divide un array de pesos en n shares de Shamir con umbral t, sobre Z_P

def split_shamir(secret: np.ndarray, n: int,
                 t: int) -> List[Tuple[int, np.ndarray]]:
    if not (1 <= t <= n):
        raise ValueError(f"Umbral t={t} inválido: debe cumplirse 1 <= t <= n={n}.")

    flat   = _quantize(secret)
    d      = len(flat)
    coeffs = [flat] + [_uniform_zp(d) for _ in range(t - 1)]

    shares: List[Tuple[int, np.ndarray]] = []
    for x_val in range(1, n + 1):
        y     = np.zeros(d, dtype=np.int64)  # Resultado de f(x_val) para cada elemento del array
        x_pow = 1  # Usamos int estándar de Python para evitar overflow
        for coeff in coeffs:
            y     = (y.astype(object) + coeff.astype(object) * x_pow) % PRIME  # f(x_val) acumulado
            x_pow = (x_pow * x_val) % PRIME  # x_val^k para el siguiente término del polinomio
            
        # Volvemos a empaquetar en int64 para eficiencia en memoria y red
        shares.append((x_val, y.astype(np.int64).reshape(secret.shape)))
    return shares

# Reconstruye el secreto a partir de t shares usando interpolación de Lagrange en Z_P
def reconstruct_shamir(shares: List[Tuple[int, np.ndarray]],
                       t: int, shape: tuple) -> np.ndarray:
    shares = shares[:t]
    xs     = [int(s[0]) for s in shares]
    ys     = [s[1].flatten() for s in shares]
    d      = len(ys[0])
    result = np.zeros(d, dtype=np.int64)

    for i in range(t):
        xi  = xs[i]
        num = 1
        den = 1
        for j in range(t):
            if i != j:
                xj  = xs[j]
                num = num * (PRIME - xj) % PRIME
                den = den * ((xi - xj) % PRIME) % PRIME
        li     = num * _modinv(den) % PRIME
        
        # Aritmética bigint de Python temporal para prevenir overflow en li * ys[i]
        result = (result.astype(object) + int(li) * ys[i].astype(object)) % PRIME
        result = result.astype(np.int64)

    return _dequantize(result, shape)


# Codifica los parámetros del modelo como shares de Shamir. Devuelve n_layers x n arrays int64 (n shares por capa).

def encode_parameters(params: List[np.ndarray], n: int,
                      t: int = None) -> List[np.ndarray]:
    t_use  = t if t is not None else (n // 2 + 1)
    
    packed: List[np.ndarray] = []
    for layer in params:
        for (_, y) in split_shamir(layer, n, t_use):
            packed.append(y)
    return packed

# Simula el intercambio P2P de shares y la suma local en cada cliente .
def simulate_p2p_exchange_and_local_sum(all_packed: List[List[np.ndarray]], n: int) -> List[List[Tuple[int, np.ndarray]]]:
    n_clients = len(all_packed)
    n_arrays  = len(all_packed[0])

    if n_arrays % n != 0:
        raise ValueError(f"Arrays por cliente ({n_arrays}) no es múltiplo de n={n}.")

    n_layers = n_arrays // n
    server_payloads = [] 

    for j in range(n):
        client_j_sum = []
        for i in range(n_layers):
            agg_j = np.zeros_like(all_packed[0][i * n + j], dtype=np.int64)
            for c in range(n_clients):
                agg_j = (agg_j.astype(object) + all_packed[c][i * n + j].astype(object)) % PRIME
                agg_j = agg_j.astype(np.int64)
            client_j_sum.append((j + 1, agg_j))
        server_payloads.append(client_j_sum)    

    return server_payloads


# El servidor central reconstruye la suma global a partir de las sumas locales pre-calculadas, sin ver los shares individuales.
def secagg_server_reconstruct(server_payloads: List[List[Tuple[int, np.ndarray]]], original_n_clients: int,t: int = None)-> List[np.ndarray]:
    n = len(server_payloads)
    # Aplicar la misma lógica robusta para el umbral por defecto
    t_use = t if t is not None else (n // 2 + 1)
    n_layers = len(server_payloads[0])
    result = []
    for i in range(n_layers):
        # La forma original se mantiene en los arrays sumados
        shape = server_payloads[0][i][1].shape
        # Agrupar los shares sumados de la capa i de todos los nodos evaluadores
        agg_shares = [client_payload[i] for client_payload in server_payloads]
        # Lagrange sobre los shares sumados -> SUM(w_i) para esta capa
        layer_sum = reconstruct_shamir(agg_shares, t_use, shape)
        # Dividir por el número original de clientes para obtener la media
        result.append(layer_sum / original_n_clients)
    return result