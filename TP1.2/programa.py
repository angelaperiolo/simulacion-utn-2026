import random
import argparse
import matplotlib.pyplot as plt

#configuracion de argumentos
parser = argparse.ArgumentParser(
    description="Simulación de estrategias de apuestas en ruleta"
)
parser.add_argument(
    "-c",
    "--corridas",
    type=int,
    required=True,
    help="Cantidad de corridas"
)
parser.add_argument(
    "-n",
    "--tiradas",
    type=int,
    required=True,
    help="Cantidad de tiradas por corrida"
)
parser.add_argument(
    "-s",
    "--sistema",
    type=str,
    required=True,
    choices=["m", "d", "f", "a"],
    help="Sistema de apuesta: m=Martingala, d=D'Alembert, f=Fibonacci, a=Apuesta fija"
)
parser.add_argument(
    "-k",
    "--capital",
    type=int,
    required=True,
    help="Capital inicial"
)
parser.add_argument(
    "-a",
    "--tipo_capital",
    type=str,
    required=True,
    choices=["f", "i"],
    help="Tipo de capital: f=finito, i=infinito"
)
args = parser.parse_args()
corridas = args.corridas
tiradas = args.tiradas
sistema = args.sistema
capital_inicial = args.capital
tipo_capital = args.tipo_capital

#configuracion general
COLOR = "red"
todos_capitales = []
todas_apuestas = []

#funcion para tirar la ruleta
def tirar_ruleta():
    numero = random.randint(0, 36)
    rojos = [
        1,3,5,7,9,12,14,16,18,
        19,21,23,25,27,30,32,34,36
    ]
    if numero in rojos:
        return "red"
    else:
        return "black"

#simulacion
for corrida in range(corridas):
    capital = capital_inicial
    capitales = [capital]
    apuestas_historial = []
    apuesta = 1
    fibonacci = [1, 1]
    indice_fib = 0
    for tirada in range(tiradas):
        #capital finito
        if tipo_capital == "f":
            if capital <= 0:
                capitales.append(capital)
                apuestas_historial.append(0)
                continue
        resultado = tirar_ruleta()
        apuesta_actual = apuesta

        #si el capital es finito no puede apostar más de lo que tiene
        if tipo_capital == "f":
            if apuesta_actual > capital:
                apuesta_actual = capital
        apuestas_historial.append(apuesta_actual)

        #gana
        if resultado == COLOR:
            capital += apuesta_actual
            #martingala
            if sistema == "m":
                apuesta = 1
            #d'alembert
            elif sistema == "d":
                if apuesta > 1:
                    apuesta -= 1
            #fibonacci
            elif sistema == "f":
                indice_fib -= 2
                if indice_fib < 0:
                    indice_fib = 0
                apuesta = fibonacci[indice_fib]
            #apuesta fija
            elif sistema == "a":
                apuesta = 1
        #pierde
        else:
            capital -= apuesta_actual
            #martingala
            if sistema == "m":
                apuesta *= 2
            #d'alembert
            elif sistema == "d":
                apuesta += 1
            #fibonacci
            elif sistema == "f":
                indice_fib += 1
                if indice_fib >= len(fibonacci):
                    fibonacci.append(
                        fibonacci[-1] + fibonacci[-2]
                    )
                apuesta = fibonacci[indice_fib]
            #apuesta fija
            elif sistema == "a":
                apuesta = 1
        capitales.append(capital)
    todos_capitales.append(capitales)
    todas_apuestas.append(apuestas_historial)

#nombre del sistema
if sistema == "m":
    nombre = "martingala"
elif sistema == "d":
    nombre = "dalembert"
elif sistema == "f":
    nombre = "fibonacci"
else:
    nombre = "apuesta_fija"

#grafico primera corrida - capital
plt.figure(figsize=(10, 5))

plt.plot(todos_capitales[0])

plt.xlabel("Tiradas")
plt.ylabel("Capital")
plt.title(f"{nombre} - Evolución del capital")
plt.grid()

plt.savefig(f"{nombre}_{tipo_capital}_corrida.png")

#grafico primera corrida - apuestas
plt.figure(figsize=(10, 5))

plt.plot(todas_apuestas[0])

plt.xlabel("Tiradas")
plt.ylabel("Valor apostado")
plt.title(f"{nombre} - Evolución de apuestas")
plt.grid()

plt.savefig(f"{nombre}_{tipo_capital}_apuestas.png")

#grafico todas las corridas
plt.figure(figsize=(10, 5))

for i in range(corridas):
    plt.plot(todos_capitales[i])
plt.xlabel("Tiradas")
plt.ylabel("Capital")
plt.title(f"{nombre} - Todas las corridas")
plt.grid()

plt.savefig(f"{nombre}_{tipo_capital}_total.png")

#mensaje
print("Simulación finalizada correctamente.")
print("Las imágenes fueron generadas.")
