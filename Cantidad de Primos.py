import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Contador de Números Primos", page_icon="🔢", layout="centered"
)

st.title("🔢 Análisis y Conteo de Números Primos")
st.write(
    "Introduce un número para comprobar si es primo, ver la lista acumulada y"
    " generar el gráfico interactivo."
)


# --- Funciones (con optimización de caché) ---
@st.cache_data
def esPrimo(n):
  if n < 2:
    return False
  if n == 2:
    return True
  m = int(np.round(np.sqrt(n)) + 1)
  for i in range(2, m):
    if n % i == 0:
      return False
  return True


@st.cache_data
def ListaPrimos(n):
  return [i for i in range(2, n + 1) if esPrimo(i)]


@st.cache_data
def CantPrimos(n):
  return len(ListaPrimos(n))


# --- Interfaz de Usuario ---
n = st.number_input(
    "Introduce el parámetro (n):",
    min_value=2,
    max_value=100000,
    value=50,
    step=1,
)

# 1. Indicar si el parámetro es primo
if esPrimo(n):
  st.success(f"¡El número **{n}** **es primo**!")
else:
  st.info(f"El número **{n}** **no es primo**.")

# 2. Lista de números primos menores o iguales a n
lista_p = ListaPrimos(n)
with st.expander(
    f"Ver lista de números primos menores o iguales a {n} ({len(lista_p)} en"
    " total)"
):
  st.write(lista_p)

# 3. Gráfico y botón de descarga
st.subheader("Gráfico de la Función Contador de Primos $\\pi(n)$")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(1, n + 1)
y = np.array([CantPrimos(i) for i in x])

ax.plot(x, y, color="blue", linewidth=2)
ax.set_title(f"Función contador de primos hasta n = {n}")
ax.set_xlabel("Números (n)")
ax.set_ylabel("Cantidad de Primos")
ax.grid(True, linestyle="--", alpha=0.6)

# Mostrar gráfico en Streamlit
st.pyplot(fig)

# Preparar el gráfico para la descarga en bytes
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

st.download_button(
    label="📥 Descargar gráfico como PNG",
    data=buf,
    file_name=f"contador_primos_{n}.png",
    mime="image/png",
)