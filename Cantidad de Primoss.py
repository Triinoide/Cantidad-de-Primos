"""
App de Streamlit: Función contador de primos π(n) y sus aproximaciones.

Ejecutar con:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:  # scipy permite calcular Li(x) de forma exacta; si no está, se integra numéricamente
    from scipy.special import expi
    HAY_SCIPY = True
except ImportError:
    HAY_SCIPY = False


# ---------------------------------------------------------------------------
# Funciones matemáticas
# ---------------------------------------------------------------------------

# Función original del proyecto (con un ajuste: 0 y 1 no son primos)
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


# Función original del proyecto
def ListaPrimos(n):
    L = []
    for i in range(2, n + 1):
        if esPrimo(i):
            L.append(i)
    return L


@st.cache_data(show_spinner=False)
def criba(n):
    """Criba de Eratóstenes: devuelve un arreglo booleano es_primo[0..n].
    Da el mismo resultado que esPrimo, pero es mucho más rápida para n grandes."""
    es_primo = np.ones(n + 1, dtype=bool)
    es_primo[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if es_primo[i]:
            es_primo[i * i :: i] = False
    return es_primo


@st.cache_data(show_spinner=False)
def contador_primos(n):
    """π(x) para x = 0, 1, ..., n (cantidad de primos menores o iguales a x)."""
    return np.cumsum(criba(n))


def factorizar(n):
    """Descomposición en factores primos: devuelve {primo: exponente}."""
    factores = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factores[d] = factores.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factores[n] = factores.get(n, 0) + 1
    return factores


def texto_factorizacion(factores):
    return " \\cdot ".join(f"{p}^{{{e}}}" if e > 1 else f"{p}" for p, e in factores.items())


def aprox_x_sobre_ln(x):
    """Aproximación de Gauss/Legendre: x / ln(x)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x >= 2, x / np.log(x), np.nan)


def aprox_legendre(x):
    """Aproximación de Legendre mejorada: x / (ln(x) - 1)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x >= 10, x / (np.log(x) - 1), np.nan)  # cerca de x=e se dispara


@st.cache_data(show_spinner=False)
def aprox_Li(n):
    """Integral logarítmica desplazada Li(x) = ∫_2^x dt / ln(t), para x = 0..n."""
    x = np.arange(n + 1, dtype=float)
    res = np.full(n + 1, np.nan)
    if n < 2:
        return res
    if HAY_SCIPY:
        res[2:] = expi(np.log(x[2:])) - expi(np.log(2.0))
    else:
        # Regla del trapecio con 20 subdivisiones por unidad
        sub = 20
        t = np.linspace(2, n, (n - 2) * sub + 1)
        f = 1 / np.log(t)
        integral = np.concatenate(([0.0], np.cumsum((f[1:] + f[:-1]) / 2 * (t[1] - t[0]))))
        res[2:] = integral[::sub]
    return res


APROXIMACIONES = {
    "x / ln(x)": {"color": "#d62728", "estilo": "--"},
    "x / (ln(x) − 1)": {"color": "#2ca02c", "estilo": "-."},
    "Li(x)": {"color": "#ff7f0e", "estilo": ":"},
}


def calcular_aprox(nombre, n):
    x = np.arange(n + 1)
    if nombre == "x / ln(x)":
        return aprox_x_sobre_ln(x)
    if nombre == "x / (ln(x) − 1)":
        return aprox_legendre(x)
    return aprox_Li(n)


def fig_a_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Contador de números primos", page_icon="🔢", layout="wide")

st.title("🔢 Función contador de primos π(n)")
st.markdown(
    "Esta aplicación cuenta cuántos números primos hay hasta un número **n** y compara "
    "esa cantidad con funciones que la aproximan, según el **Teorema de los Números Primos**."
)

# ----- Barra lateral -----
with st.sidebar:
    st.header("⚙️ Configuración")
    N = st.number_input(
        "Graficar hasta n =", min_value=10, max_value=2_000_000, value=1000, step=100,
        help="Límite superior del gráfico.",
    )
    N = int(N)

    aprox_elegidas = st.multiselect(
        "Aproximaciones a comparar",
        list(APROXIMACIONES.keys()),
        default=["x / ln(x)", "Li(x)"],
    )

    st.subheader("Opciones del gráfico")
    escala_log = st.checkbox("Escala logarítmica en el eje x", value=False)
    mostrar_grilla = st.checkbox("Mostrar grilla", value=True)
    marcar_numero = st.checkbox("Marcar en el gráfico el número consultado", value=True)

    st.divider()
    st.caption(
        "Para n grandes se usa la *Criba de Eratóstenes*, que da el mismo resultado que "
        "la función `esPrimo` del proyecto pero mucho más rápido."
    )

# ----- Verificador de primalidad -----
st.subheader("🔍 ¿Es primo?")
col_in, col_res = st.columns([1, 2])
with col_in:
    numero = int(
        st.number_input("Ingresá un número natural", min_value=1, value=97, step=1, key="numero")
    )

with col_res:
    if esPrimo(numero):
        st.success(f"✅ **{numero}** es un número **primo**.")
    elif numero == 1:
        st.info("ℹ️ **1** no es primo ni compuesto (por definición, un primo tiene exactamente dos divisores).")
    else:
        fac = factorizar(numero)
        menor_divisor = min(fac)
        st.error(f"❌ **{numero}** **no** es primo: es divisible por {menor_divisor}.")
        st.latex(f"{numero} = {texto_factorizacion(fac)}")

# Primo anterior y siguiente
if numero <= 2_000_000:
    anterior = next((k for k in range(numero - 1, 1, -1) if esPrimo(k)), None)
    siguiente = next(k for k in range(numero + 1, numero + 10_000) if esPrimo(k))
    c1, c2, c3 = st.columns(3)
    c1.metric("Primo anterior", anterior if anterior else "—")
    c2.metric("Primo siguiente", siguiente)
    if numero <= N:
        c3.metric(f"π({numero})", int(contador_primos(N)[numero]),
                  help="Cantidad de primos menores o iguales al número ingresado.")
    else:
        c3.metric(f"π({numero})", int(contador_primos(numero)[numero]))

st.divider()

# ----- Cálculos principales -----
with st.spinner("Calculando primos..."):
    pi = contador_primos(N)
    x = np.arange(N + 1)
    valores_aprox = {nombre: calcular_aprox(nombre, N) for nombre in aprox_elegidas}

# ----- Métricas en n = N -----
st.subheader(f"📊 Resultados para n = {N:,}".replace(",", "."))
cols = st.columns(1 + len(aprox_elegidas))
cols[0].metric("π(n) real", f"{int(pi[N]):,}".replace(",", "."))
for col, nombre in zip(cols[1:], aprox_elegidas):
    valor = valores_aprox[nombre][N]
    error_rel = (valor - pi[N]) / pi[N] * 100
    col.metric(nombre, f"{valor:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."),
               delta=f"{error_rel:+.2f} % de error".replace(".", ","), delta_color="off")

# ----- Pestañas -----
tab_graf, tab_error, tab_tabla, tab_teoria = st.tabs(
    ["📈 Gráfico", "📉 Error de las aproximaciones", "📋 Tabla y lista de primos", "📚 Teoría"]
)

with tab_graf:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.step(x[1:], pi[1:], where="post", color="blue", linewidth=1.8, label="π(x) (real)")
    for nombre in aprox_elegidas:
        est = APROXIMACIONES[nombre]
        ax.plot(x[2:], valores_aprox[nombre][2:], est["estilo"], color=est["color"],
                linewidth=1.6, label=nombre)
    if marcar_numero and 1 <= numero <= N:
        ax.plot(numero, pi[numero], "o", color="black", markersize=7, zorder=5)
        ax.annotate(f"π({numero}) = {pi[numero]}", (numero, pi[numero]),
                    textcoords="offset points", xytext=(10, -15), fontsize=10)
    ax.set_title("Función contador de primos y sus aproximaciones")
    ax.set_xlabel("Números (x)")
    ax.set_ylabel("Cantidad de primos ≤ x")
    if escala_log:
        ax.set_xscale("log")
    ax.grid(mostrar_grilla, alpha=0.4)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.download_button(
        "⬇️ Descargar gráfico (PNG)", data=fig_a_png(fig),
        file_name=f"contador_primos_hasta_{N}.png", mime="image/png",
    )
    plt.close(fig)

with tab_error:
    if not aprox_elegidas:
        st.info("Elegí al menos una aproximación en la barra lateral.")
    else:
        st.markdown(
            "El cociente **π(x) / aproximación** debería acercarse a **1** a medida que x crece "
            "(eso es lo que afirma el Teorema de los Números Primos)."
        )
        inicio = 10
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for nombre in aprox_elegidas:
            est = APROXIMACIONES[nombre]
            with np.errstate(divide="ignore", invalid="ignore"):
                cociente = pi[inicio:] / valores_aprox[nombre][inicio:]
            ax2.plot(x[inicio:], cociente, color=est["color"], linewidth=1.4, label=nombre)
        ax2.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax2.set_title("Cociente π(x) / aproximación")
        ax2.set_xlabel("Números (x)")
        ax2.set_ylabel("π(x) / aproximación")
        if escala_log:
            ax2.set_xscale("log")
        ax2.grid(mostrar_grilla, alpha=0.4)
        ax2.legend()
        st.pyplot(fig2)
        st.download_button(
            "⬇️ Descargar gráfico de error (PNG)", data=fig_a_png(fig2),
            file_name=f"error_aproximaciones_hasta_{N}.png", mime="image/png",
        )
        plt.close(fig2)

with tab_tabla:
    # Tabla en potencias de 10 (y N)
    puntos = sorted({10 ** k for k in range(1, 8) if 10 ** k <= N} | {N})
    filas = []
    for p in puntos:
        fila = {"n": p, "π(n)": int(pi[p])}
        for nombre in aprox_elegidas:
            v = valores_aprox[nombre][p]
            fila[nombre] = round(float(v), 2)
            fila[f"Error % {nombre}"] = round(float((v - pi[p]) / pi[p] * 100), 2)
        filas.append(fila)
    df = pd.DataFrame(filas)
    st.markdown("**Comparación en distintos valores de n**")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="comparacion_primos.csv", mime="text/csv")

    st.markdown(f"**Lista de primos menores o iguales a {N}** ({int(pi[N])} primos)")
    primos = np.nonzero(criba(N))[0]
    if len(primos) <= 2000:
        st.text(", ".join(map(str, primos)))
    else:
        st.caption("Se muestran los primeros 2000; descargá el archivo para verlos todos.")
        st.text(", ".join(map(str, primos[:2000])) + ", ...")
    st.download_button("⬇️ Descargar lista de primos (TXT)", "\n".join(map(str, primos)),
                       file_name=f"primos_hasta_{N}.txt", mime="text/plain")

with tab_teoria:
    st.markdown(
        r"""
### La función contador de primos
Se define $\pi(x)$ como la cantidad de números primos menores o iguales a $x$.
Por ejemplo, $\pi(10) = 4$, ya que los primos hasta 10 son 2, 3, 5 y 7.

### El Teorema de los Números Primos
Conjeturado por **Gauss** y **Legendre** a fines del siglo XVIII y demostrado en 1896 por
**Hadamard** y **de la Vallée Poussin**, afirma que

$$\lim_{x \to \infty} \frac{\pi(x)}{x / \ln x} = 1$$

es decir, para $x$ grande, $\pi(x) \approx \dfrac{x}{\ln x}$.

### Las aproximaciones
| Aproximación | Comentario |
|---|---|
| $\dfrac{x}{\ln x}$ | La más simple. Siempre queda **por debajo** de $\pi(x)$ (para $x \ge 17$). |
| $\dfrac{x}{\ln x - 1}$ | Corrección propuesta por Legendre; mucho más precisa. |
| $\mathrm{Li}(x) = \displaystyle\int_2^x \frac{dt}{\ln t}$ | Propuesta por Gauss; es la mejor de las tres. |

### Una interpretación
Cerca de un número $x$, la "probabilidad" de que un número elegido al azar sea primo es
aproximadamente $\dfrac{1}{\ln x}$: los primos se vuelven cada vez más escasos, pero nunca se terminan
(como ya demostró Euclides).
"""
    )
