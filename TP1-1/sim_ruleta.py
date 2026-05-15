import random
import argparse
import statistics
import matplotlib.pyplot as plt

#configuracion de argumentos
parser = argparse.ArgumentParser(description="Simulacion de una ruleta")
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
    "-e",
    "--elegido",
    type=int,
    required=True,
    help="Numero elegido"
)

args = parser.parse_args()

cantidad_corridas = args.corridas
cantidad_tiradas = args.tiradas
numero_elegido = args.elegido

#valores teoricos
frecuencia_esperada = 1 / 37
media_esperada = 18
varianza_esperada = 114

#desvio estandar teorico
numeros_ruleta = list(range(37))
desvio_esperado = statistics.pstdev(numeros_ruleta)

#listas generales
frecuencias_corridas = []
medias_corridas = []
varianzas_corridas = []
desvios_corridas = []

#simulacion
for corrida in range(cantidad_corridas):
    resultados = []
    frecuencias = []
    medias = []
    varianzas = []
    desvios = []

    apariciones = 0

    for tirada in range(1, cantidad_tiradas + 1):
        numero = random.randint(0, 36)
        resultados.append(numero)

        #frecuencia relativa
        if numero == numero_elegido:
            apariciones += 1

        frecuencia_relativa = apariciones / tirada
        frecuencias.append(frecuencia_relativa)

        #media
        media_actual = statistics.mean(resultados)
        medias.append(media_actual)

        #varianza
        if len(resultados) > 1:
            varianza_actual = statistics.pvariance(resultados)
            desvio_actual = statistics.pstdev(resultados)
        else:
            varianza_actual = 0
            desvio_actual = 0

        varianzas.append(varianza_actual)
        desvios.append(desvio_actual)

    frecuencias_corridas.append(frecuencias)
    medias_corridas.append(medias)
    varianzas_corridas.append(varianzas)
    desvios_corridas.append(desvios)

#eje X
x = list(range(1, cantidad_tiradas + 1))

#graficos PRIMERA CORRIDA
#frecuencia
plt.figure(figsize=(10, 5))
plt.plot(x, frecuencias_corridas[0], label="Frecuencia relativa")
plt.axhline(frecuencia_esperada, linestyle="--", label="Valor esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Frecuencia relativa")
plt.title("Frecuencia relativa vs numero de tiradas")
plt.legend()
plt.grid()
plt.savefig("frecuencia_corrida1.png")

#media
plt.figure(figsize=(10, 5))
plt.plot(x, medias_corridas[0], label="Media")
plt.axhline(media_esperada, linestyle="--", label="Valor esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Media")
plt.title("Media vs numero de tiradas")
plt.legend()
plt.grid()
plt.savefig("media_corrida1.png")

#varianza
plt.figure(figsize=(10, 5))
plt.plot(x, varianzas_corridas[0], label="Varianza")
plt.axhline(varianza_esperada, linestyle="--", label="Valor esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Varianza")
plt.title("Varianza vs numero de tiradas")
plt.legend()
plt.grid()
plt.savefig("varianza_corrida1.png")

#desvio
plt.figure(figsize=(10, 5))
plt.plot(x, desvios_corridas[0], label="Desvio estandar")
plt.axhline(desvio_esperado, linestyle="--", label="Valor esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Desvio estandar")
plt.title("Desvio estandar vs numero de tiradas")
plt.legend()
plt.grid()
plt.savefig("desvio_corrida1.png")

#graficos TODAS las corridas
#frecuencias
plt.figure(figsize=(10, 5))

for i in range(cantidad_corridas):
    plt.plot(x, frecuencias_corridas[i], label=f"Corrida {i+1}")

plt.axhline(frecuencia_esperada, linestyle="--", color="black", label="Esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Frecuencia relativa")
plt.title("Frecuencia relativa de todas las corridas")
plt.legend()
plt.grid()
plt.savefig("frecuencia_total.png")

#medias
plt.figure(figsize=(10, 5))

for i in range(cantidad_corridas):
    plt.plot(x, medias_corridas[i], label=f"Corrida {i+1}")

plt.axhline(media_esperada, linestyle="--", color="black", label="Esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Media")
plt.title("Media de todas las corridas")
plt.legend()
plt.grid()
plt.savefig("media_total.png")

#varianzas
plt.figure(figsize=(10, 5))

for i in range(cantidad_corridas):
    plt.plot(x, varianzas_corridas[i], label=f"Corrida {i+1}")

plt.axhline(varianza_esperada, linestyle="--", color="black", label="Esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Varianza")
plt.title("Varianza de todas las corridas")
plt.legend()
plt.grid()
plt.savefig("varianza_total.png")

#desvios
plt.figure(figsize=(10, 5))

for i in range(cantidad_corridas):
    plt.plot(x, desvios_corridas[i], label=f"Corrida {i+1}")

plt.axhline(desvio_esperado, linestyle="--", color="black", label="Esperado")
plt.xlabel("Cantidad de tiradas")
plt.ylabel("Desvio estandar")
plt.title("Desvio estandar de todas las corridas")
plt.legend()
plt.grid()
plt.savefig("desvio_total.png")

#mensaje final
print("Simulacion finalizada correctamente.")
print("Las imagenes fueron generadas en el directorio actual.")
