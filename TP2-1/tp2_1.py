import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

#generadores
def cuadrados_medios(seed, n):
    numeros = []
    x = seed
    for _ in range(n):
        cuadrado = str(x ** 2).zfill(8)
        x = int(cuadrado[2:6])
        numeros.append(x / 10000)
    return numeros

def gcl(seed, a, c, m, n):
    numeros = []
    x = seed
    for _ in range(n):
        x = (a * x + c) % m
        numeros.append(x / m)
    return numeros

def python_random(n):
    return [random.random() for _ in range(n)]

#test de media
def test_media(datos):
    return np.mean(datos)

#test de varianza
def test_varianza(datos):
    return np.var(datos)

#test chi-cuadrado
def test_chi_cuadrado(datos, intervalos=10):
    frecuencias, _ = np.histogram(datos, bins=intervalos, range=(0, 1))
    esperadas = [len(datos) / intervalos] * intervalos
    chi2, pvalue = chisquare(
        f_obs=frecuencias,
        f_exp=esperadas
    )
    return chi2, pvalue

#test de corridas
def test_corridas(datos):
    media = np.mean(datos)
    simbolos = []
    for x in datos:
        if x >= media:
            simbolos.append(1)
        else:
            simbolos.append(0)
    corridas = 1
    for i in range(1, len(simbolos)):
        if simbolos[i] != simbolos[i - 1]:
            corridas += 1
    return corridas

#histograma
def histograma(datos, titulo):
    plt.figure(figsize=(8, 5))
    plt.hist(datos, bins=10)
    plt.title(titulo)
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.show()

#dispersion Ui vs Ui+1
def dispersion(datos, titulo):
    x = datos[:-1]
    y = datos[1:]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5)
    plt.title(titulo)
    plt.xlabel("Ui")
    plt.ylabel("Ui+1")
    plt.show()

#main
N = 10000
# Cuadrados medios
cm = cuadrados_medios(
    seed=5735,
    n=N
)

#GCL
gcl_nums = gcl(
    seed=12345,
    a=1664525,
    c=1013904223,
    m=2**32,
    n=N
)

#python
py = python_random(N)
generadores = {
    "Cuadrados Medios": cm,
    "GCL": gcl_nums,
    "Python": py
}

print("\nRESULTADOS\n")
for nombre, datos in generadores.items():
    media = test_media(datos)
    varianza = test_varianza(datos)
    chi2, pvalue = test_chi_cuadrado(datos)
    corridas = test_corridas(datos)
    print("=" * 50)
    print(nombre)
    print("=" * 50)
    print(f"Media      : {media:.6f}")
    print(f"Varianza   : {varianza:.6f}")
    print(f"Chi²       : {chi2:.6f}")
    print(f"P-Value    : {pvalue:.6f}")
    print(f"Corridas   : {corridas}")
    print()
    histograma(datos, f"Histograma - {nombre}")
    dispersion(datos, f"Dispersión - {nombre}")
