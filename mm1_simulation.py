"""
==================================================================
TP3 - Simulacion de un modelo M/M/1 por eventos discretos
==================================================================

Implementa:
  - Simulacion de una corrida individual M/M/1 (o M/M/1/K con capacidad
    finita) mediante el metodo de proximo evento (next-event time advance).
  - Corrida de multiples replicas (minimo 30) y promediado de resultados.
  - Formulas teoricas de M/M/1 y M/M/1/K para comparar contra la simulacion.
  - Barrido de experimentos: tasas de arribo al 25/50/75/100/125% de la
    tasa de servicio, y capacidades de cola finitas (0, 2, 5, 10, 50).
  - Generacion de tablas (CSV) y graficos (PNG) listos para incluir en el
    informe de LaTeX.

Uso rapido (linea de comandos), para poder variar parametros en clase:

    # Una corrida "manual" con parametros propios
    python mm1_simulation.py --lambda_ 0.8 --mu 1.0 --sim_time 20000 --reps 30

    # Con cola finita (capacidad total = servidor + cola = 6, o sea
    # tamanio de cola de espera = 5)
    python mm1_simulation.py --lambda_ 0.9 --mu 1.0 --capacity 6 --reps 30

    # Correr todo el barrido de experimentos que pide el TP
    python mm1_simulation.py --experiment
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

OUT_DIR = "resultados_mm1"


# ==================================================================
# 1. SIMULACION POR EVENTOS DISCRETOS (una sola corrida)
# ==================================================================
def simulate_mm1(lmbda, mu, sim_time, capacity=None, seed=None, n_snapshots=200):
    """
    Simula una cola M/M/1 (o M/M/1/K si `capacity` no es None) por eventos
    discretos, usando el metodo de proximo evento.

    Parametros
    ----------
    lmbda    : tasa de arribos (clientes por unidad de tiempo)
    mu       : tasa de servicio (clientes por unidad de tiempo)
    sim_time : tiempo total de simulacion (tiempo simulado, no real)
    capacity : capacidad TOTAL del sistema (servidor + cola). None = infinita
    seed     : semilla del generador aleatorio (reproducibilidad)
    n_snapshots : cantidad de puntos guardados para graficar la evolucion
                  temporal de L(t) y Lq(t)

    Devuelve
    --------
    dict con las metricas de la corrida (ver claves al final de la funcion).
    """
    rng = np.random.default_rng(seed)

    clock = 0.0
    last_event_time = 0.0
    n_system = 0
    server_busy = False

    next_arrival = rng.exponential(1.0 / lmbda)
    next_departure = math.inf

    system_customers = deque()  # clientes actualmente en el sistema (FIFO)

    area_system = 0.0   # integral de n(t) dt  -> usado para L
    area_queue = 0.0    # integral de nq(t) dt -> usado para Lq
    busy_time = 0.0      # tiempo con servidor ocupado -> utilizacion

    state_time_queue = {}  # tiempo acumulado en cada estado "clientes en cola"

    snapshot_times = np.linspace(0, sim_time, n_snapshots)
    snap_idx = 0
    ts_t, ts_L, ts_Lq = [], [], []

    n_arrivals = 0
    n_accepted = 0
    n_blocked = 0
    n_served = 0
    times_in_system = []
    times_in_queue = []

    while True:
        clock = min(next_arrival, next_departure)
        if clock >= sim_time:
            clock = sim_time

        dt = clock - last_event_time
        area_system += n_system * dt
        q_len = n_system - (1 if server_busy else 0)
        area_queue += q_len * dt
        if server_busy:
            busy_time += dt
        state_time_queue[q_len] = state_time_queue.get(q_len, 0.0) + dt

        while snap_idx < n_snapshots and snapshot_times[snap_idx] <= clock:
            t_snap = snapshot_times[snap_idx]
            ts_t.append(t_snap)
            ts_L.append(area_system / t_snap if t_snap > 0 else 0.0)
            ts_Lq.append(area_queue / t_snap if t_snap > 0 else 0.0)
            snap_idx += 1

        last_event_time = clock
        if clock >= sim_time:
            break

        if next_arrival <= next_departure:
            # ---- evento: arribo ----
            n_arrivals += 1
            if capacity is not None and n_system >= capacity:
                n_blocked += 1  # cliente rechazado (cola llena)
            else:
                n_accepted += 1
                n_system += 1
                cust = {"arrival": clock, "service_start": None}
                system_customers.append(cust)
                if not server_busy:
                    server_busy = True
                    cust["service_start"] = clock
                    next_departure = clock + rng.exponential(1.0 / mu)
            next_arrival = clock + rng.exponential(1.0 / lmbda)
        else:
            # ---- evento: salida (fin de servicio) ----
            n_served += 1
            n_system -= 1
            cust = system_customers.popleft()
            times_in_system.append(clock - cust["arrival"])
            times_in_queue.append(cust["service_start"] - cust["arrival"])
            if system_customers:
                nxt = system_customers[0]
                nxt["service_start"] = clock
                next_departure = clock + rng.exponential(1.0 / mu)
            else:
                server_busy = False
                next_departure = math.inf

    L = area_system / sim_time
    Lq = area_queue / sim_time
    rho_sim = busy_time / sim_time
    W = float(np.mean(times_in_system)) if times_in_system else 0.0
    Wq = float(np.mean(times_in_queue)) if times_in_queue else 0.0
    blocking_prob = n_blocked / n_arrivals if n_arrivals > 0 else 0.0
    queue_dist = {k: v / sim_time for k, v in sorted(state_time_queue.items())}

    return {
        "L": L, "Lq": Lq, "W": W, "Wq": Wq, "rho": rho_sim,
        "n_arrivals": n_arrivals, "n_accepted": n_accepted,
        "n_blocked": n_blocked, "n_served": n_served,
        "blocking_prob": blocking_prob,
        "queue_dist": queue_dist,
        "ts_t": np.array(ts_t), "ts_L": np.array(ts_L), "ts_Lq": np.array(ts_Lq),
    }


# ==================================================================
# 2. MULTIPLES REPLICAS (minimo 30 corridas por experimento)
# ==================================================================
def run_replications(lmbda, mu, sim_time, capacity=None, n_reps=30, seed0=0):
    """Corre `n_reps` replicas independientes y promedia los resultados."""
    rows = []
    all_queue_dists = []
    ts_L_all, ts_Lq_all = [], []
    ts_t = None

    for i in range(n_reps):
        res = simulate_mm1(lmbda, mu, sim_time, capacity=capacity, seed=seed0 + i)
        rows.append({
            "rep": i, "L": res["L"], "Lq": res["Lq"], "W": res["W"], "Wq": res["Wq"],
            "rho": res["rho"], "blocking_prob": res["blocking_prob"],
            "n_arrivals": res["n_arrivals"], "n_blocked": res["n_blocked"],
        })
        all_queue_dists.append(res["queue_dist"])
        ts_L_all.append(res["ts_L"])
        ts_Lq_all.append(res["ts_Lq"])
        ts_t = res["ts_t"]

    df = pd.DataFrame(rows)

    max_k = max(max(d.keys()) for d in all_queue_dists)
    dist_matrix = np.zeros((n_reps, max_k + 1))
    for i, d in enumerate(all_queue_dists):
        for k, p in d.items():
            dist_matrix[i, k] = p
    queue_dist_mean = dist_matrix.mean(axis=0)

    ts_L_mean = np.mean(ts_L_all, axis=0)
    ts_Lq_mean = np.mean(ts_Lq_all, axis=0)

    summary_mean = df.drop(columns="rep").mean()
    summary_ci95 = 1.96 * df.drop(columns="rep").std(ddof=1) / np.sqrt(n_reps)

    return {
        "df": df, "summary_mean": summary_mean, "summary_ci95": summary_ci95,
        "queue_dist_mean": queue_dist_mean,
        "ts_t": ts_t, "ts_L_mean": ts_L_mean, "ts_Lq_mean": ts_Lq_mean,
    }


# ==================================================================
# 3. VALORES TEORICOS (M/M/1 y M/M/1/K)
# ==================================================================
def mm1_theoretical(lmbda, mu):
    """Formulas clasicas de M/M/1 (capacidad infinita). Requiere rho < 1."""
    rho = lmbda / mu
    if rho >= 1:
        return {"rho": rho, "L": math.inf, "Lq": math.inf,
                "W": math.inf, "Wq": math.inf}
    L = rho / (1 - rho)
    Lq = rho ** 2 / (1 - rho)
    W = 1 / (mu - lmbda)
    Wq = lmbda / (mu * (mu - lmbda))
    return {"rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq}


def queue_length_theoretical(rho, max_k):
    """
    Distribucion teorica del numero de clientes EN COLA (sin contar al
    que esta siendo atendido) para M/M/1 con capacidad infinita.
    Se obtiene a partir de P(n en sistema) = (1-rho) rho^n:
        P(Nq=0) = P(N=0) + P(N=1) = 1 - rho^2
        P(Nq=k) = P(N=k+1) = (1-rho) rho^(k+1),  k >= 1

    Para rho >= 1 el sistema es inestable y no existe distribucion
    estacionaria: se devuelve un arreglo de NaN (no corresponde graficar
    ni comparar contra la simulacion en ese caso).
    """
    if rho >= 1:
        return np.full(max_k + 1, np.nan)
    p = np.zeros(max_k + 1)
    p[0] = 1 - rho ** 2
    for k in range(1, max_k + 1):
        p[k] = (1 - rho) * rho ** (k + 1)
    return p


def mm1k_theoretical(lmbda, mu, K):
    """
    Formulas de M/M/1/K. K = capacidad TOTAL del sistema (servidor + cola),
    es decir, si el tamanio de la cola de espera es Kq, entonces K = Kq + 1.
    """
    rho = lmbda / mu
    n = np.arange(0, K + 1)
    if abs(rho - 1.0) < 1e-9:
        Pn = np.ones(K + 1) / (K + 1)
    else:
        Pn = (1 - rho) * rho ** n / (1 - rho ** (K + 1))
    P_blocking = Pn[-1]
    lambda_eff = lmbda * (1 - P_blocking)
    L = float(np.sum(n * Pn))
    rho_server = 1 - Pn[0]
    Lq = L - rho_server
    W = L / lambda_eff if lambda_eff > 0 else math.inf
    Wq = Lq / lambda_eff if lambda_eff > 0 else math.inf
    return {"rho": rho, "Pn": Pn, "P_blocking": float(P_blocking), "L": L,
            "Lq": Lq, "W": W, "Wq": Wq, "lambda_eff": lambda_eff}


# ==================================================================
# 4. EXPERIMENTOS DEL TP
# ==================================================================
def experimento_variar_rho(mu=1.0, sim_time=20000, n_reps=30,
                            factores=(0.25, 0.50, 0.75, 1.00, 1.25)):
    """
    Varia la tasa de arribo como 25/50/75/100/125% de la tasa de servicio,
    con capacidad infinita, y compara simulacion vs teoria.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    filas = []
    curvas_L = {}

    for f in factores:
        lmbda = f * mu
        print(f"Corriendo rho={f:.2f} (lambda={lmbda:.3f}, mu={mu}) "
              f"con {n_reps} replicas...")
        res = run_replications(lmbda, mu, sim_time, capacity=None, n_reps=n_reps)
        teo = mm1_theoretical(lmbda, mu)

        filas.append({
            "factor_rho": f, "lambda": lmbda, "mu": mu,
            "L_sim": res["summary_mean"]["L"], "L_ci95": res["summary_ci95"]["L"],
            "L_teo": teo["L"],
            "Lq_sim": res["summary_mean"]["Lq"], "Lq_ci95": res["summary_ci95"]["Lq"],
            "Lq_teo": teo["Lq"],
            "W_sim": res["summary_mean"]["W"], "W_ci95": res["summary_ci95"]["W"],
            "W_teo": teo["W"],
            "Wq_sim": res["summary_mean"]["Wq"], "Wq_ci95": res["summary_ci95"]["Wq"],
            "Wq_teo": teo["Wq"],
            "rho_sim": res["summary_mean"]["rho"], "rho_teo": teo["rho"],
        })
        curvas_L[f] = (res["ts_t"], res["ts_L_mean"], res["ts_Lq_mean"])

        # distribucion de clientes en cola para este rho
        # (se acota a 60 estados: alcanza de sobra para graficar, y evita
        # arrastrar soportes enormes en el caso inestable rho>=1)
        dist_full = res["queue_dist_mean"]
        max_len = min(len(dist_full), 60)
        dist = dist_full[:max_len]
        teo_dist = queue_length_theoretical(f, max_len - 1)
        pd.DataFrame({
            "n_en_cola": np.arange(len(dist)),
            "prob_sim": dist,
            "prob_teo": teo_dist,
        }).to_csv(f"{OUT_DIR}/dist_cola_rho{f:.2f}.csv", index=False)
        _plot_queue_distribution(dist, teo_dist, f)

    df = pd.DataFrame(filas)
    df.to_csv(f"{OUT_DIR}/resumen_variar_rho.csv", index=False)
    print(df.to_string(index=False))

    # ---- graficos ----
    _plot_metric_vs_rho(df, "L", "Numero promedio de clientes en el sistema (L)")
    _plot_metric_vs_rho(df, "Lq", "Numero promedio de clientes en cola (Lq)")
    _plot_metric_vs_rho(df, "W", "Tiempo promedio en el sistema (W)")
    _plot_metric_vs_rho(df, "Wq", "Tiempo promedio en cola (Wq)")
    _plot_evolucion_temporal(curvas_L)

    return df


def _plot_queue_distribution(dist, teo_dist, factor_rho, max_bars=15):
    n_max = min(len(dist), max_bars)
    x = np.arange(n_max)
    ancho = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - ancho / 2, dist[:n_max], width=ancho, label="Simulado")
    if not np.isnan(teo_dist[:n_max]).all():
        plt.bar(x + ancho / 2, teo_dist[:n_max], width=ancho, label="Teorico")
    plt.xlabel("Clientes en cola (n)")
    plt.ylabel("Probabilidad")
    plt.title(fr"Distribucion de clientes en cola ($\rho$={factor_rho:.2f})")
    plt.xticks(x)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dist_cola_rho{factor_rho:.2f}.png", dpi=150)
    plt.close()


def _plot_metric_vs_rho(df, metric, titulo):
    plt.figure(figsize=(6, 4))
    plt.errorbar(df["factor_rho"], df[f"{metric}_sim"], yerr=df[f"{metric}_ci95"],
                 fmt="o-", capsize=4, label="Simulacion (IC 95%)")
    plt.plot(df["factor_rho"], df[f"{metric}_teo"], "s--", label="Teorico")
    plt.xlabel(r"$\lambda / \mu$")
    plt.ylabel(metric)
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{metric}_vs_rho.png", dpi=150)
    plt.close()


def _plot_evolucion_temporal(curvas_L):
    plt.figure(figsize=(7, 5))
    for f, (t, L, Lq) in curvas_L.items():
        plt.plot(t, L, label=fr"$\rho$={f:.2f}")
    plt.xlabel("Tiempo de simulacion")
    plt.ylabel("L(t) acumulado")
    plt.title("Convergencia de L(t) segun tiempo de simulacion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/evolucion_L_vs_tiempo.png", dpi=150)
    plt.close()


def experimento_cola_finita(mu=1.0, sim_time=20000, n_reps=30,
                             factores=(0.25, 0.50, 0.75, 1.00, 1.25),
                             tamanios_cola=(0, 2, 5, 10, 50)):
    """
    Calcula la probabilidad de denegacion de servicio (bloqueo) para
    distintos tamanios de cola de espera (0, 2, 5, 10, 50) y distintas
    tasas de arribo, comparando simulacion vs formula de M/M/1/K.

    Nota: tamanio_cola = espacios de ESPERA. La capacidad total del
    sistema usada en la simulacion es capacity = tamanio_cola + 1
    (1 lugar corresponde al cliente en servicio).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    filas = []

    for f in factores:
        lmbda = f * mu
        for Kq in tamanios_cola:
            K = Kq + 1  # capacidad total del sistema
            print(f"rho={f:.2f}, tamanio de cola={Kq} (capacidad total={K})...")
            res = run_replications(lmbda, mu, sim_time, capacity=K, n_reps=n_reps)
            teo = mm1k_theoretical(lmbda, mu, K)
            filas.append({
                "factor_rho": f, "lambda": lmbda, "mu": mu,
                "tamanio_cola": Kq, "capacidad_total": K,
                "P_bloqueo_sim": res["summary_mean"]["blocking_prob"],
                "P_bloqueo_ci95": res["summary_ci95"]["blocking_prob"],
                "P_bloqueo_teo": teo["P_blocking"],
            })

    df = pd.DataFrame(filas)
    df.to_csv(f"{OUT_DIR}/resumen_cola_finita.csv", index=False)
    print(df.to_string(index=False))

    plt.figure(figsize=(7, 5))
    for f in factores:
        sub = df[df["factor_rho"] == f]
        plt.plot(sub["tamanio_cola"], sub["P_bloqueo_sim"], "o-",
                  label=fr"$\rho$={f:.2f} (sim)")
        plt.plot(sub["tamanio_cola"], sub["P_bloqueo_teo"], "x--",
                  color=plt.gca().lines[-1].get_color())
    plt.xlabel("Tamanio de la cola (espacios de espera)")
    plt.ylabel("Probabilidad de denegacion de servicio")
    plt.title("Probabilidad de bloqueo vs tamanio de cola")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bloqueo_vs_tamanio_cola.png", dpi=150)
    plt.close()

    return df


# ==================================================================
# 5. LINEA DE COMANDOS
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulacion M/M/1 - TP3")
    parser.add_argument("--lambda_", type=float, default=0.8,
                         help="Tasa de arribos (clientes/unidad de tiempo)")
    parser.add_argument("--mu", type=float, default=1.0,
                         help="Tasa de servicio (clientes/unidad de tiempo)")
    parser.add_argument("--sim_time", type=float, default=20000,
                         help="Tiempo total de simulacion por corrida")
    parser.add_argument("--capacity", type=int, default=None,
                         help="Capacidad total del sistema (servidor+cola). "
                              "Vacio = infinita")
    parser.add_argument("--reps", type=int, default=30,
                         help="Cantidad de replicas (minimo 30)")
    parser.add_argument("--experiment", action="store_true",
                         help="Corre el barrido completo de experimentos del TP")
    args = parser.parse_args()

    if args.experiment:
        experimento_variar_rho(mu=args.mu, sim_time=args.sim_time, n_reps=args.reps)
        experimento_cola_finita(mu=args.mu, sim_time=args.sim_time, n_reps=args.reps)
        print(f"\nListo. Resultados en la carpeta '{OUT_DIR}/'")
        return

    # ---- corrida individual con los parametros indicados ----
    res = run_replications(args.lambda_, args.mu, args.sim_time,
                            capacity=args.capacity, n_reps=args.reps)
    print(f"\n=== Resultados ({args.reps} replicas) ===")
    print(res["summary_mean"].to_string())
    print("\n=== Intervalo de confianza 95% (semiancho) ===")
    print(res["summary_ci95"].to_string())

    if args.capacity is None:
        teo = mm1_theoretical(args.lambda_, args.mu)
        print("\n=== Valores teoricos M/M/1 ===")
        for k, v in teo.items():
            print(f"{k}: {v:.4f}")
    else:
        teo = mm1k_theoretical(args.lambda_, args.mu, args.capacity)
        print("\n=== Valores teoricos M/M/1/K ===")
        print(f"P_blocking: {teo['P_blocking']:.4f}, L: {teo['L']:.4f}, "
              f"Lq: {teo['Lq']:.4f}, W: {teo['W']:.4f}, Wq: {teo['Wq']:.4f}")


if __name__ == "__main__":
    main()
