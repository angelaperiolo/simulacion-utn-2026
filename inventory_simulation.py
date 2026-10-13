"""
==================================================================
TP3 - Simulacion de un modelo de Inventario (politica (s,S)) por
      revision periodica.
==================================================================

Se elige una politica de revision periodica (s, S):
  - Cada periodo (por ejemplo, cada dia) se observa la posicion de
    inventario (stock disponible + pedidos en camino - pedidos
    pendientes/backorders).
  - Si esa posicion cae por debajo del punto de pedido `s`, se emite
    una orden de reposicion hasta el nivel `S` (orden-hasta-el-nivel S).
  - La orden tarda `lead_time` periodos en llegar.
  - La demanda de cada periodo es aleatoria (Poisson), y el faltante
    se modela con backorders (se puede cambiar a ventas perdidas).

Esta es la politica de inventario estocastica mas usada como caso de
estudio en los libros de simulacion (Banks, Law & Kelton, Sheldon
Ross), y es la contraparte practica del modelo determinista EOQ con
faltantes planificados, que se usa aqui como VALOR TEORICO de
referencia (formulas de Wilson con faltantes).

Justificacion de los parametros por defecto (ver `DEFAULTS` abajo):
  - Demanda ~ Poisson(lambda=5) por dia: representa una demanda
    minorista moderada, con variabilidad realista (var = media, como
    corresponde a un proceso de arribos tipo Poisson).
  - Lead time = 3 dias: tiempo de reposicion tipico de un proveedor
    local.
  - Costo de orden K = 50 (fijo por pedido, independiente de la
    cantidad): agrupa gastos administrativos/transporte.
  - Costo de mantenimiento h = 1 por unidad y por dia: costo de
    oportunidad de capital inmovilizado + espacio de deposito.
  - Costo de faltante p = 5 por unidad y por dia en backorder: se fija
    en 5 veces el costo de mantenimiento para reflejar que la falta de
    stock (venta perdida o cliente disconforme) es mas cara que
    mantener inventario de mas; es un valor tipico usado en la
    bibliografia para forzar politicas con nivel de servicio alto.
  - s y S NO se fijan arbitrariamente: se calculan con la funcion
    `suggest_sS`, que combina el lote economico con faltantes (EOQ) y
    un stock de seguridad para un nivel de servicio del 95% (z=1.65)
    durante el lead time. Esto es lo que se usa como punto de partida
    "razonado" y luego se explora su sensibilidad con un barrido.

Uso rapido (linea de comandos), para variar parametros en clase:

    # Corrida individual con la politica sugerida automaticamente
    python inventory_simulation.py --reps 30

    # Corrida con (s,S) elegidos a mano
    python inventory_simulation.py --s 20 --S 60 --reps 30

    # Barrido completo de experimentos del TP (busqueda de la mejor
    # politica (s,S) + comparacion contra el valor teorico EOQ)
    python inventory_simulation.py --experiment
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

OUT_DIR = "resultados_inventario"

DEFAULTS = dict(
    lambda_demand=5.0,   # demanda media por periodo (Poisson)
    lead_time=3,         # periodos hasta que llega un pedido
    K=50.0,              # costo fijo por orden
    c=0.0,               # costo variable por unidad pedida (0 = no influye
                          # en la politica optima, se deja disponible por si
                          # se quiere agregar costo de compra)
    h=1.0,               # costo de mantenimiento por unidad y por periodo
    p=5.0,               # costo de faltante por unidad y por periodo
    T=2000,              # cantidad de periodos simulados por corrida
    reps=30,             # replicas minimas pedidas por el TP
)


# ==================================================================
# 1. SIMULACION DE UNA CORRIDA (politica (s,S), revision periodica)
# ==================================================================
def simulate_inventory(T, lambda_demand, lead_time, s, S, K, c, h, p,
                        backorder=True, seed=None):
    """
    Simula T periodos de un sistema de inventario (s,S) con revision
    periodica, demanda Poisson y backorders (o ventas perdidas si
    backorder=False).

    Devuelve un dict con los costos totales, sus tasas por periodo, y
    la evolucion temporal de cada componente (para graficar
    convergencia).
    """
    rng = np.random.default_rng(seed)

    on_hand = S            # se asume que el sistema arranca lleno (nivel S)
    backorders = 0.0        # unidades pendientes de entregar (si backorder=True)
    on_order = deque()      # pedidos en camino: (periodo_de_llegada, cantidad)

    total_order_cost = 0.0
    total_holding_cost = 0.0
    total_shortage_cost = 0.0
    num_orders = 0
    total_units_ordered = 0
    total_demand = 0.0
    total_lost_sales = 0.0

    hist_t = np.arange(1, T + 1)
    hist_order_cost = np.zeros(T)
    hist_holding_cost = np.zeros(T)
    hist_shortage_cost = np.zeros(T)
    hist_inventory = np.zeros(T)

    for t in range(1, T + 1):
        # 1. recibir pedidos que llegan en este periodo
        while on_order and on_order[0][0] == t:
            _, qty = on_order.popleft()
            on_hand += qty
            if backorder and backorders > 0:
                cubierto = min(backorders, on_hand)
                backorders -= cubierto
                on_hand -= cubierto

        # 2. demanda del periodo
        demand = rng.poisson(lambda_demand)
        total_demand += demand

        if backorder:
            if on_hand >= demand:
                on_hand -= demand
            else:
                faltante = demand - on_hand
                on_hand = 0
                backorders += faltante
        else:
            vendido = min(on_hand, demand)
            perdido = demand - vendido
            on_hand -= vendido
            total_lost_sales += perdido

        # 3. costos del periodo (mantenimiento y faltante)
        holding_cost_t = h * on_hand
        shortage_cost_t = p * backorders if backorder else p * perdido

        total_holding_cost += holding_cost_t
        total_shortage_cost += shortage_cost_t

        # 4. revisar politica (s,S) sobre la posicion de inventario
        pedidos_en_camino = sum(q for _, q in on_order)
        inventory_position = on_hand - backorders + pedidos_en_camino
        order_cost_t = 0.0
        if inventory_position <= s:
            qty = S - inventory_position
            if qty > 0:
                order_cost_t = K + c * qty
                on_order.append((t + lead_time, qty))
                num_orders += 1
                total_units_ordered += qty
        total_order_cost += order_cost_t

        hist_order_cost[t - 1] = total_order_cost
        hist_holding_cost[t - 1] = total_holding_cost
        hist_shortage_cost[t - 1] = total_shortage_cost
        hist_inventory[t - 1] = on_hand - backorders

    total_cost = total_order_cost + total_holding_cost + total_shortage_cost

    return {
        "total_order_cost": total_order_cost,
        "total_holding_cost": total_holding_cost,
        "total_shortage_cost": total_shortage_cost,
        "total_cost": total_cost,
        "order_cost_rate": total_order_cost / T,
        "holding_cost_rate": total_holding_cost / T,
        "shortage_cost_rate": total_shortage_cost / T,
        "total_cost_rate": total_cost / T,
        "num_orders": num_orders,
        "avg_inventory": float(np.mean(hist_inventory)),
        "fill_rate": 1 - (total_lost_sales / total_demand if (not backorder and total_demand > 0) else 0.0),
        "hist_t": hist_t,
        "hist_order_cost": hist_order_cost,
        "hist_holding_cost": hist_holding_cost,
        "hist_shortage_cost": hist_shortage_cost,
        "hist_total_cost": hist_order_cost + hist_holding_cost + hist_shortage_cost,
    }


# ==================================================================
# 2. MULTIPLES REPLICAS (minimo 30 corridas por experimento)
# ==================================================================
def run_replications_inventory(T, lambda_demand, lead_time, s, S, K, c, h, p,
                                backorder=True, n_reps=30, seed0=0):
    rows = []
    hist_total_cost_rate_all = []
    hist_t = None

    for i in range(n_reps):
        res = simulate_inventory(T, lambda_demand, lead_time, s, S, K, c, h, p,
                                  backorder=backorder, seed=seed0 + i)
        rows.append({
            "rep": i,
            "order_cost_rate": res["order_cost_rate"],
            "holding_cost_rate": res["holding_cost_rate"],
            "shortage_cost_rate": res["shortage_cost_rate"],
            "total_cost_rate": res["total_cost_rate"],
            "num_orders": res["num_orders"],
            "avg_inventory": res["avg_inventory"],
        })
        hist_t = res["hist_t"]
        # tasa acumulada de costo total en el tiempo (para grafico de convergencia)
        hist_total_cost_rate_all.append(res["hist_total_cost"] / hist_t)

    df = pd.DataFrame(rows)
    summary_mean = df.drop(columns="rep").mean()
    summary_ci95 = 1.96 * df.drop(columns="rep").std(ddof=1) / np.sqrt(n_reps)
    ts_total_cost_rate_mean = np.mean(hist_total_cost_rate_all, axis=0)

    return {
        "df": df, "summary_mean": summary_mean, "summary_ci95": summary_ci95,
        "hist_t": hist_t, "ts_total_cost_rate_mean": ts_total_cost_rate_mean,
    }


# ==================================================================
# 3. VALORES TEORICOS: EOQ con faltantes planificados (modelo de
#    Wilson determinista) y regla de (s,S) sugerida
# ==================================================================
def eoq_shortage_theoretical(D, K, h, p):
    """
    Lote economico (EOQ) con faltantes planificados, demanda
    determinista D (unidades por periodo). Sirve como cota/referencia
    teorica para el sistema estocastico (s,S), ya que en el limite de
    demanda con poca variabilidad ambos deberian aproximarse.
    """
    Q_star = math.sqrt(2 * D * K / h * (h + p) / p)
    S_star = Q_star * p / (h + p)          # inventario maximo tras reponer
    max_shortage = Q_star - S_star          # faltante maximo permitido
    cost_rate = math.sqrt(2 * D * K * h * p / (h + p))  # costo total / periodo
    order_cost_rate = D / Q_star * K
    holding_cost_rate = h * S_star ** 2 / (2 * Q_star)
    shortage_cost_rate = p * max_shortage ** 2 / (2 * Q_star)
    return {
        "Q_star": Q_star, "S_star": S_star, "max_shortage": max_shortage,
        "order_cost_rate": order_cost_rate,
        "holding_cost_rate": holding_cost_rate,
        "shortage_cost_rate": shortage_cost_rate,
        "total_cost_rate": cost_rate,
    }


def suggest_sS(lambda_demand, lead_time, K, h, p, service_z=1.65):
    """
    Propone valores de (s,S) combinando:
      - El lote economico con faltantes (Q*) de la formula de Wilson,
        usado como tamanio de lote de reposicion.
      - Un stock de seguridad = z * desvio estandar de la demanda
        durante el lead time, para un nivel de servicio del ~95%
        (z=1.65), sumado a la demanda esperada durante el lead time
        para fijar el punto de pedido s.
    """
    eoq = eoq_shortage_theoretical(lambda_demand, K, h, p)
    mean_lt_demand = lambda_demand * lead_time
    std_lt_demand = math.sqrt(lambda_demand * lead_time)  # Poisson: var = media
    safety_stock = service_z * std_lt_demand
    s = mean_lt_demand + safety_stock
    S = s + eoq["Q_star"]
    return {"s": round(s), "S": round(S), "Q_star": eoq["Q_star"],
            "safety_stock": safety_stock, "eoq": eoq}


# ==================================================================
# 4. EXPERIMENTOS DEL TP
# ==================================================================
def experimento_baseline(params=DEFAULTS, n_reps=30):
    """
    Corre la politica (s,S) sugerida automaticamente, compara contra
    el valor teorico EOQ y grafica la evolucion temporal de los
    costos (convergencia de las tasas de costo por periodo).
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    sug = suggest_sS(params["lambda_demand"], params["lead_time"],
                      params["K"], params["h"], params["p"])
    s, S = sug["s"], sug["S"]
    print(f"Politica sugerida: s={s}, S={S} (Q*={sug['Q_star']:.2f}, "
          f"stock de seguridad={sug['safety_stock']:.2f})")

    res = run_replications_inventory(
        params["T"], params["lambda_demand"], params["lead_time"], s, S,
        params["K"], params["c"], params["h"], params["p"],
        backorder=True, n_reps=n_reps)

    teo = sug["eoq"]

    tabla = pd.DataFrame({
        "componente": ["Costo de orden", "Costo de mantenimiento",
                        "Costo de faltante", "Costo total"],
        "simulacion (por periodo)": [
            res["summary_mean"]["order_cost_rate"],
            res["summary_mean"]["holding_cost_rate"],
            res["summary_mean"]["shortage_cost_rate"],
            res["summary_mean"]["total_cost_rate"],
        ],
        "IC95 (semiancho)": [
            res["summary_ci95"]["order_cost_rate"],
            res["summary_ci95"]["holding_cost_rate"],
            res["summary_ci95"]["shortage_cost_rate"],
            res["summary_ci95"]["total_cost_rate"],
        ],
        "teorico EOQ (por periodo)": [
            teo["order_cost_rate"], teo["holding_cost_rate"],
            teo["shortage_cost_rate"], teo["total_cost_rate"],
        ],
    })
    tabla.to_csv(f"{OUT_DIR}/baseline_s{s}_S{S}.csv", index=False)
    print(tabla.to_string(index=False))

    # grafico de convergencia de la tasa de costo total por periodo
    plt.figure(figsize=(7, 5))
    plt.plot(res["hist_t"], res["ts_total_cost_rate_mean"],
              label="Costo total / periodo (simulado)")
    plt.axhline(teo["total_cost_rate"], color="red", linestyle="--",
                 label="Costo total / periodo (teorico EOQ)")
    plt.xlabel("Periodo de simulacion")
    plt.ylabel("Costo promedio acumulado por periodo")
    plt.title(f"Convergencia del costo total (s={s}, S={S})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergencia_costo_s{s}_S{S}.png", dpi=150)
    plt.close()

    return tabla, res


def experimento_barrido_sS(params=DEFAULTS, n_reps=30,
                            s_values=None, S_deltas=None):
    """
    Barre distintas combinaciones de (s, S) para comparar politicas y
    encontrar (aproximadamente) la de menor costo total simulado. Se
    varia s alrededor del stock de seguridad sugerido, y S como
    s + delta, con delta tomando distintos multiplos de Q*.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    sug = suggest_sS(params["lambda_demand"], params["lead_time"],
                      params["K"], params["h"], params["p"])
    s_base, Q_star = sug["s"], sug["Q_star"]

    if s_values is None:
        s_values = [max(0, round(s_base * f)) for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
    if S_deltas is None:
        S_deltas = [round(Q_star * f) for f in (0.5, 0.75, 1.0, 1.25, 1.5)]

    filas = []
    for s in s_values:
        for delta in S_deltas:
            S = s + delta
            res = run_replications_inventory(
                params["T"], params["lambda_demand"], params["lead_time"], s, S,
                params["K"], params["c"], params["h"], params["p"],
                backorder=True, n_reps=n_reps)
            filas.append({
                "s": s, "S": S, "Q": delta,
                "order_cost_rate": res["summary_mean"]["order_cost_rate"],
                "holding_cost_rate": res["summary_mean"]["holding_cost_rate"],
                "shortage_cost_rate": res["summary_mean"]["shortage_cost_rate"],
                "total_cost_rate": res["summary_mean"]["total_cost_rate"],
                "total_cost_ci95": res["summary_ci95"]["total_cost_rate"],
            })
            print(f"s={s:>4}, S={S:>4} (Q={delta:>4}) -> "
                  f"costo total/periodo = {filas[-1]['total_cost_rate']:.2f}")

    df = pd.DataFrame(filas)
    df.to_csv(f"{OUT_DIR}/barrido_sS.csv", index=False)

    mejor = df.loc[df["total_cost_rate"].idxmin()]
    print("\nMejor politica encontrada en el barrido:")
    print(mejor.to_string())

    # heatmap de costo total en funcion de (s, S)
    pivot = df.pivot(index="s", columns="S", values="total_cost_rate")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis_r",
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.max(), pivot.index.min()])
    plt.colorbar(label="Costo total / periodo")
    plt.xlabel("S")
    plt.ylabel("s")
    plt.title("Costo total promedio segun politica (s, S)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/heatmap_sS.png", dpi=150)
    plt.close()

    # curvas de costo total vs S, una linea por cada valor de s
    plt.figure(figsize=(7, 5))
    for s in s_values:
        sub = df[df["s"] == s].sort_values("S")
        plt.errorbar(sub["S"], sub["total_cost_rate"], yerr=sub["total_cost_ci95"],
                      marker="o", capsize=3, label=f"s={s}")
    plt.xlabel("S")
    plt.ylabel("Costo total / periodo")
    plt.title("Costo total vs nivel de reposicion S, para distintos s")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/costo_vs_S.png", dpi=150)
    plt.close()

    return df, mejor


# ==================================================================
# 5. LINEA DE COMANDOS
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulacion de Inventario (s,S) - TP3")
    parser.add_argument("--lambda_demand", type=float, default=DEFAULTS["lambda_demand"],
                         help="Demanda media por periodo (Poisson)")
    parser.add_argument("--lead_time", type=int, default=DEFAULTS["lead_time"],
                         help="Tiempo de espera de reposicion (en periodos)")
    parser.add_argument("--s", type=float, default=None,
                         help="Punto de pedido. Si no se indica, se sugiere automaticamente")
    parser.add_argument("--S", type=float, default=None,
                         help="Nivel de reposicion. Si no se indica, se sugiere automaticamente")
    parser.add_argument("--K", type=float, default=DEFAULTS["K"], help="Costo fijo por orden")
    parser.add_argument("--c", type=float, default=DEFAULTS["c"], help="Costo variable por unidad")
    parser.add_argument("--h", type=float, default=DEFAULTS["h"],
                         help="Costo de mantenimiento por unidad y periodo")
    parser.add_argument("--p", type=float, default=DEFAULTS["p"],
                         help="Costo de faltante por unidad y periodo")
    parser.add_argument("--T", type=int, default=DEFAULTS["T"], help="Cantidad de periodos por corrida")
    parser.add_argument("--reps", type=int, default=DEFAULTS["reps"], help="Cantidad de replicas (minimo 30)")
    parser.add_argument("--lost_sales", action="store_true",
                         help="Usar ventas perdidas en vez de backorders")
    parser.add_argument("--experiment", action="store_true",
                         help="Corre el barrido completo de experimentos del TP")
    args = parser.parse_args()

    params = dict(lambda_demand=args.lambda_demand, lead_time=args.lead_time,
                  K=args.K, c=args.c, h=args.h, p=args.p, T=args.T, reps=args.reps)

    if args.experiment:
        experimento_baseline(params=params, n_reps=args.reps)
        experimento_barrido_sS(params=params, n_reps=args.reps)
        print(f"\nListo. Resultados en la carpeta '{OUT_DIR}/'")
        return

    if args.s is None or args.S is None:
        sug = suggest_sS(args.lambda_demand, args.lead_time, args.K, args.h, args.p)
        s = args.s if args.s is not None else sug["s"]
        S = args.S if args.S is not None else sug["S"]
        print(f"(s,S) sugerido automaticamente: s={s}, S={S}")
    else:
        s, S = args.s, args.S

    res = run_replications_inventory(
        args.T, args.lambda_demand, args.lead_time, s, S,
        args.K, args.c, args.h, args.p,
        backorder=not args.lost_sales, n_reps=args.reps)

    print(f"\n=== Resultados ({args.reps} replicas, s={s}, S={S}) ===")
    print(res["summary_mean"].to_string())
    print("\n=== Intervalo de confianza 95% (semiancho) ===")
    print(res["summary_ci95"].to_string())

    teo = eoq_shortage_theoretical(args.lambda_demand, args.K, args.h, args.p)
    print("\n=== Valores teoricos EOQ con faltantes ===")
    for k, v in teo.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
