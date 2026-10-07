
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def _normalize_and_validate_shares(
    shares: ArrayLike, 
    atol: float = 1e-4
) -> np.ndarray:
    """
    Valida y normaliza el vector de cuotas de mercado a escala decimal [0, 1].

    Parámetros:
    -----------
    shares : ArrayLike
        Cuotas individuales de mercado en escala [0, 1] o [0, 100].
    atol : float
        Tolerancia absoluta para absorber imprecisiones de coma flotante (IEEE 754).

    Retorna:
    --------
    np.ndarray : Vector 1D en formato flotante normalizado en el rango [0, 1].
    """
    s = np.asarray(shares, dtype=np.float64).flatten()

    if s.size == 0:
        raise ValueError("El array de cuotas de mercado no puede estar vacío.")

    # 1. Detección automática de escala según la magnitud de la suma
    total_sum = float(np.sum(s))
    
    is_percentage = np.isclose(total_sum, 100.0, atol=atol)
    is_decimal = np.isclose(total_sum, 1.0, atol=atol)

    if not (is_percentage or is_decimal):
        raise ValueError(
            f"La suma de cuotas ({total_sum:.6f}) no equivale al total del mercado. "
            "Debe sumar exactamente 1.0 (escala decimal) o 100.0 (porcentaje)."
        )

    # Convertir a escala decimal uniforme [0, 1]
    if is_percentage:
        s = s / 100.0

    # 2. Validación de rangos admisibles [0, 1]
    # Se admite una pequeña tolerancia negativa/positiva por precisión numérica
    if np.any(s < -atol) or np.any(s > 1.0 + atol):
        raise ValueError(
            "Existen cuotas individuales fuera del dominio válido [0, 1] o [0, 100%]."
        )

    # Acotar estrictamente dentro de [0, 1] para eliminar ruido numérico
    s = np.clip(s, 0.0, 1.0)

    # Re-normalizar exactamente para garantizar suma = 1.0
    s_sum = np.sum(s)
    if s_sum > 0:
        s = s / s_sum

    return s


def concentration_ratio(shares: ArrayLike, k: int = 4) -> float:
    """
    Calcula el Ratio de Concentración CR_k (suma de las k mayores cuotas).

    Parámetros:
    -----------
    shares : ArrayLike
        Cuotas de mercado (decimales o porcentajes).
    k : int
        Número de firmas líderes a considerar (comúnmente k=4 o k=8).

    Retorna:
    --------
    float : Proporción de mercado controlada por las k mayores firmas en [0, 1].
    """
    if k <= 0:
        raise ValueError("El parámetro 'k' debe ser un entero estrictamente positivo.")

    s = _normalize_and_validate_shares(shares)
    
    # Ordenar de mayor a menor y tomar los primeros k elementos
    sorted_shares = np.sort(s)[::-1]
    return float(np.sum(sorted_shares[:k]))


def herfindahl_hirschman_index(shares: ArrayLike, scale_points: bool = True) -> float:
    """
    Calcula el Índice de Herfindahl-Hirschman (HHI).

    Parámetros:
    -----------
    shares : ArrayLike
        Cuotas de mercado (decimales o porcentajes).
    scale_points : bool, por defecto True
        Si es True, devuelve la escala regulatoria estándar [0, 10000].
        Si es False, devuelve en escala decimal [0, 1].

    Retorna:
    --------
    float : Valor del HHI.
    """
    s = _normalize_and_validate_shares(shares)
    hhi_decimal = float(np.sum(s ** 2))

    return hhi_decimal * 10000.0 if scale_points else hhi_decimal


def dominance_index(shares: ArrayLike) -> float:
    """
    Calcula el Índice de Dominancia (ID) de García Alba Idunate (1990).
    Mide el grado de asimetría relativa y el peso del líder dentro de la estructura del HHI:
        ID = sum( (s_i^2 / sum(s_j^2))^2 ) = sum( h_i^2 )

    Retorna:
    --------
    float : Valor acotado en [1/n, 1], donde 1 indica dominancia absoluta.
    """
    s = _normalize_and_validate_shares(shares)
    squared_shares = s ** 2
    sum_squared = np.sum(squared_shares)

    if sum_squared == 0:
        return 0.0

    # Cuota relativa de cada firma dentro del HHI (h_i)
    h = squared_shares / sum_squared
    return float(np.sum(h ** 2))


def entropy_index(shares: ArrayLike, base: float = np.e) -> float:
    """
    Calcula el Índice de Entropía de Theil (IE): -sum(s_i * log(s_i)).
    Aplica la convención de teoría de la información: 0 * ln(0) = 0.

    Parámetros:
    -----------
    shares : ArrayLike
        Cuotas de mercado (decimales o porcentajes).
    base : float, por defecto e (logaritmo natural)
        Base logarítmica. Puede ajustarse a 2 (bits) o 10.

    Retorna:
    --------
    float : Entropía del mercado (mayor valor implica menor concentración).
    """
    s = _normalize_and_validate_shares(shares)

    # Filtrar cuotas estrictamente positivas para evitar indeterminación de log(0)
    positive_s = s[s > 0.0]

    if positive_s.size == 0:
        return 0.0

    log_s = np.log(positive_s) if base == np.e else np.log(positive_s) / np.log(base)
    return float(-np.sum(positive_s * log_s))
    import numpy as np
from typing import Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]


def force_normalize_shares(values: ArrayLike) -> np.ndarray:
    """
    Normaliza cualquier vector de métricas de tamaño de mercado (ingresos, 
    unidades vendidas o cuotas con error de redondeo) dividiendo cada 
    elemento por la suma total.

    Parámetros:
    -----------
    values : ArrayLike
        Vector de valores no negativos.

    Retorna:
    --------
    np.ndarray : Vector 1D en escala decimal [0, 1] cuya suma es exactamente 1.0.
    """
    arr = np.asarray(values, dtype=np.float64).flatten()

    if arr.size == 0:
        raise ValueError("El vector de valores no puede estar vacío.")

    # Validar que no contenga valores estrictamente negativos
    if np.any(arr < 0.0):
        raise ValueError("No se permiten valores negativos en las cuotas o métricas de mercado.")

    total = np.sum(arr)
    if np.isclose(total, 0.0, atol=1e-12):
        raise ValueError("La suma total del mercado es cero o indeterminada; no es posible normalizar.")

    # División por la suma total
    normalized = arr / total

    # Re-proyección estricta para garantizar suma idéntica a 1.0 y rango [0, 1]
    return np.clip(normalized, 0.0, 1.0)


def _validate_and_prepare_shares(
    shares: ArrayLike, 
    atol: float = 1e-4, 
    auto_normalize: bool = True
) -> np.ndarray:
    """
    Valida las cuotas con margen de tolerancia numérica y las prepara para el cálculo.

    Parámetros:
    -----------
    shares : ArrayLike
        Cuotas en escala [0, 1] o [0, 100].
    atol : float, por defecto 1e-4 (0.01%)
        Tolerancia absoluta admitida para desviaciones de redondeo en la suma y límites.
    auto_normalize : bool, por defecto True
        Si es True, absorbe el residuo numérico dividiendo por la suma real para
        devolver cuotas cuya suma interna sea idéntica a 1.0.

    Retorna:
    --------
    np.ndarray : Vector 1D validado y normalizado a escala [0, 1].
    """
    s = np.asarray(shares, dtype=np.float64).flatten()

    if s.size == 0:
        raise ValueError("El array de cuotas de mercado no puede estar vacío.")

    total_sum = float(np.sum(s))

    # Detección de escala con tolerancia absoluta
    is_percentage = np.isclose(total_sum, 100.0, atol=atol)
    is_decimal = np.isclose(total_sum, 1.0, atol=atol)

    if not (is_percentage or is_decimal):
        raise ValueError(
            f"La suma de cuotas ({total_sum:.8f}) difiere de 1.0 (o 100%) más allá "
            f"de la tolerancia permitida (±{atol}). "
            "Usa 'force_normalize_shares' si deseas forzar la escala."
        )

    # Convertir a escala decimal
    if is_percentage:
        s = s / 100.0
        # Escalar la tolerancia relativa a la base decimal
        atol = atol / 100.0

    # Validación de límites individuales con tolerancia
    if np.any(s < -atol) or np.any(s > 1.0 + atol):
        raise ValueError(
            f"Existen cuotas individuales fuera del intervalo admisible [-{atol}, 1 + {atol}]."
        )

    # Eliminar posibles valores infinitesimales como -1e-17 debidos a operaciones previas
    s = np.clip(s, 0.0, 1.0)

    # Eliminar el remanente de punto flotante para que la suma sea exactamente 1.0
    if auto_normalize:
        s = s / np.sum(s)

    return s
    import numpy as np
from typing import Literal, Optional


IndicatorType = Literal["CRk", "IHH", "ID", "IE"]


def simulate_market_concentration_monte_carlo(
    n_firms: int,
    n_iterations: int = 1000,
    indicator: IndicatorType = "IHH",
    k: int = 4,
    hhi_points: bool = True,
    entropy_base: float = np.e,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Ejecuta una simulación de Monte Carlo para calcular la distribución muestral
    de un índice de concentración de mercado bajo escenarios estocásticos homogéneos.

    Justificación de la Distribución:
    --------------------------------
    Se utiliza una distribución de Dirichlet simétrica con parámetro alpha = 1.
    La densidad conjunta en el símplex unitario Delta^(N-1) = {s en R^N : sum(s_i) = 1, s_i >= 0}
    es uniforme y constante f(s) = (N-1)!, garantizando que ningún competidor tenga sesgo
    a priori y que la suma sea exactamente 1.0 en cada escenario generado.

    Parámetros:
    -----------
    n_firms : int
        Número de firmas en el mercado (N >= 2).
    n_iterations : int, por defecto 1000
        Número de simulaciones de Monte Carlo a ejecutar.
    indicator : {"CRk", "IHH", "ID", "IE"}, por defecto "IHH"
        Índice a calcular:
        - "CRk": Ratio de concentración de las k mayores firmas.
        - "IHH": Índice de Herfindahl-Hirschman.
        - "ID" : Índice de Dominancia (García Alba Idunate).
        - "IE" : Índice de Entropía de Theil.
    k : int, por defecto 4
        Número de firmas para el cálculo del CRk (requerido si indicator="CRk").
    hhi_points : bool, por defecto True
        Si True, escala el IHH a base 10.000 puntos. Si False, escala [0, 1].
    entropy_base : float, por defecto e (logaritmo natural)
        Base para el cálculo del Índice de Entropía.
    random_state : Optional[int], por defecto None
        Semilla para el generador pseudoaleatorio de NumPy (reproducibilidad).

    Retorna:
    --------
    np.ndarray : Vector 1D de longitud `n_iterations` con el valor del índice en cada simulación.
    """
    if n_firms < 2:
        raise ValueError("El número de firmas (n_firms) debe ser mayor o igual a 2.")
    if n_iterations <= 0:
        raise ValueError("El número de iteraciones debe ser un entero estrictamente positivo.")

    # 1. Configurar generador de números aleatorios moderno de NumPy
    rng = np.random.default_rng(random_state)

    # 2. Generación vectorizada de la matriz de cuotas: Shape = (n_iterations, n_firms)
    # alpha = ones(n_firms) asegura muestreo uniforme en el símplex
    alpha = np.ones(n_firms, dtype=np.float64)
    shares_matrix = rng.dirichlet(alpha, size=n_iterations)

    # Re-normalización estricta por filas para disipar cualquier artefacto residual IEEE 754
    shares_matrix = shares_matrix / shares_matrix.sum(axis=1, keepdims=True)

    # 3. Cálculo vectorizado según el indicador seleccionado
    ind = indicator.upper()

    if ind == "CRK":
        effective_k = min(k, n_firms)
        # np.partition es O(N) promedio, mucho más rápido que sort completo O(N log N)
        # Obtiene las k mayores cuotas por fila
        top_k_elements = np.partition(shares_matrix, n_firms - effective_k, axis=1)[:, n_firms - effective_k:]
        return np.sum(top_k_elements, axis=1)

    elif ind == "IHH":
        # Suma de cuadrados por fila: sum_i (s_i^2)
        sum_sq = np.sum(shares_matrix ** 2, axis=1)
        return sum_sq * 10000.0 if hhi_points else sum_sq

    elif ind == "ID":
        # ID = sum_i ( (s_i^2 / sum_j s_j^2)^2 ) = sum_i (h_i^2)
        sq = shares_matrix ** 2
        sum_sq = np.sum(sq, axis=1, keepdims=True)
        h = sq / sum_sq
        return np.sum(h ** 2, axis=1)

    elif ind == "IE":
        # IE = -sum_i (s_i * ln(s_i)) con convención 0 * ln(0) = 0
        # Reemplazar ceros marginales por 1 antes del logaritmo (1 * ln(1) = 0 no altera la suma)
        safe_shares = np.where(shares_matrix > 0.0, shares_matrix, 1.0)
        log_shares = np.log(safe_shares) if entropy_base == np.e else np.log(safe_shares) / np.log(entropy_base)
        
        # Anular explícitamente donde la cuota era 0
        elementwise_entropy = np.where(shares_matrix > 0.0, shares_matrix * log_shares, 0.0)
        return -np.sum(elementwise_entropy, axis=1)

    else:
        raise ValueError(f"Indicador desconocido '{indicator}'. Opciones válidas: 'CRk', 'IHH', 'ID', 'IE'.")
        import warnings
from typing import Tuple, Optional


class LatencyWarning(UserWarning):
    """Advertencia emitida cuando el volumen de simulación puede degradar la interactividad web."""
    pass


def validate_monte_carlo_iterations(
    n_iterations: int,
    warn_threshold: int = 3000,
    min_iterations: int = 100,
    max_iterations: int = 10000
) -> Tuple[int, Optional[str]]:
    """
    Valida y acota el número de iteraciones para simulaciones de Monte Carlo en entornos web.

    Parámetros:
    -----------
    n_iterations : int
        Número de iteraciones solicitadas por el cliente.
    warn_threshold : int, por defecto 3000
        Límite a partir del cual se emite una advertencia de latencia/memoria.
    min_iterations : int, por defecto 100
        Límite inferior para garantizar significancia estadística.
    max_iterations : int, por defecto 10000
        Límite superior duro para prevenir sobrecarga de CPU y consumo de memoria.

    Retorna:
    --------
    Tuple[int, Optional[str]]:
        - int: Número de iteraciones validado.
        - Optional[str]: Mensaje de advertencia si superó `warn_threshold`, o None si está en rango óptimo.

    Lanza:
    ------
    TypeError: Si n_iterations no es un entero.
    ValueError: Si n_iterations está fuera del intervalo [min_iterations, max_iterations].
    """
    if not isinstance(n_iterations, (int, np.integer)):
        raise TypeError(f"El número de iteraciones debe ser un entero, recibido: {type(n_iterations).__name__}.")

    # Validación de límites duros
    if n_iterations < min_iterations or n_iterations > max_iterations:
        raise ValueError(
            f"El número de iteraciones ({n_iterations}) está fuera del rango permitido "
            f"[{min_iterations}, {max_iterations}]. Ajusta el parámetro para continuar."
        )

    warning_msg = None

    # Evaluación de umbral de latencia para la interfaz web
    if n_iterations > warn_threshold:
        warning_msg = (
            f"Aviso de rendimiento: Has solicitado {n_iterations} iteraciones (> {warn_threshold}). "
            "El cálculo generará matrices de memoria más densas y un incremento en el tiempo "
            "de respuesta de la interfaz o latencia de red."
        )
        warnings.warn(warning_msg, category=LatencyWarning, stacklevel=2)

    return int(n_iterations), warning_msg
    import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Union, Literal, Tuple, Optional

ArrayLike = Union[Sequence[float], np.ndarray]
IndicatorType = Literal["CRk", "IHH", "ID", "IE"]

# ==============================================================================
# 1. FUNCIONES MATEMÁTICAS Y VALIDACIÓN
# ==============================================================================

def force_normalize_shares(values: ArrayLike) -> np.ndarray:
    """Normaliza un vector de cuotas dividiendo por su suma total."""
    arr = np.asarray(values, dtype=np.float64).flatten()
    if arr.size == 0:
        raise ValueError("El vector de valores no puede estar vacío.")
    if np.any(arr < 0.0):
        raise ValueError("No se permiten valores negativos en las cuotas de mercado.")
    total = np.sum(arr)
    if np.isclose(total, 0.0, atol=1e-12):
        raise ValueError("La suma total es cero; no es posible normalizar.")
    return np.clip(arr / total, 0.0, 1.0)


def _validate_and_prepare_shares(shares: ArrayLike, atol: float = 1e-4) -> np.ndarray:
    """Valida tolerancias de coma flotante y escala decimal o porcentual."""
    s = np.asarray(shares, dtype=np.float64).flatten()
    if s.size == 0:
        raise ValueError("El array de cuotas no puede estar vacío.")

    total_sum = float(np.sum(s))
    is_percentage = np.isclose(total_sum, 100.0, atol=atol)
    is_decimal = np.isclose(total_sum, 1.0, atol=atol)

    if not (is_percentage or is_decimal):
        raise ValueError(
            f"La suma ({total_sum:.4f}) no equivale al 100% o 1.0 (±{atol}). "
            "Normaliza los datos antes de continuar."
        )

    if is_percentage:
        s = s / 100.0
        atol = atol / 100.0

    if np.any(s < -atol) or np.any(s > 1.0 + atol):
        raise ValueError("Existen cuotas individuales fuera del intervalo admisible [0, 1].")

    s = np.clip(s, 0.0, 1.0)
    return s / np.sum(s)


def concentration_ratio(shares: ArrayLike, k: int = 4) -> float:
    """Calcula CR_k (suma de las k mayores firmas)."""
    s = _validate_and_prepare_shares(shares)
    k_eff = min(k, len(s))
    return float(np.sum(np.sort(s)[::-1][:k_eff]))


def herfindahl_hirschman_index(shares: ArrayLike, scale_points: bool = True) -> float:
    """Calcula el IHH en base 10.000 puntos o escala decimal."""
    s = _validate_and_prepare_shares(shares)
    hhi_dec = float(np.sum(s ** 2))
    return hhi_dec * 10000.0 if scale_points else hhi_dec


def dominance_index(shares: ArrayLike) -> float:
    """Calcula el Índice de Dominancia de García Alba Idunate."""
    s = _validate_and_prepare_shares(shares)
    sq = s ** 2
    sum_sq = np.sum(sq)
    if sum_sq == 0:
        return 0.0
    h = sq / sum_sq
    return float(np.sum(h ** 2))


def entropy_index(shares: ArrayLike) -> float:
    """Calcula el Índice de Entropía de Theil resolviendo 0 * ln(0) = 0."""
    s = _validate_and_prepare_shares(shares)
    pos_s = s[s > 0.0]
    if pos_s.size == 0:
        return 0.0
    return float(-np.sum(pos_s * np.log(pos_s)))


def calculate_indicator(shares: ArrayLike, indicator: IndicatorType, k: int = 4) -> float:
    """Despachador unificado para calcular el índice respectivo."""
    if indicator == "CRk":
        return concentration_ratio(shares, k=k)
    elif indicator == "IHH":
        return herfindahl_hirschman_index(shares, scale_points=True)
    elif indicator == "ID":
        return dominance_index(shares)
    elif indicator == "IE":
        return entropy_index(shares)
    raise ValueError(f"Indicador desconocido: {indicator}")

# ==============================================================================
# 2. SIMULACIÓN DE MONTE CARLO VECTORIZADA
# ==============================================================================

def simulate_market_concentration_monte_carlo(
    n_firms: int,
    n_iterations: int = 1000,
    indicator: IndicatorType = "IHH",
    k: int = 4,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Simulación vectorizada vía Dirichlet simétrica (alpha=1).
    Garantiza muestreo uniforme y homogéneo en el símplex unitario.
    """
    rng = np.random.default_rng(random_state)
    alpha = np.ones(n_firms, dtype=np.float64)
    shares_matrix = rng.dirichlet(alpha, size=n_iterations)
    shares_matrix = shares_matrix / shares_matrix.sum(axis=1, keepdims=True)

    ind = indicator.upper()
    if ind == "CRK":
        effective_k = min(k, n_firms)
        top_k = np.partition(shares_matrix, n_firms - effective_k, axis=1)[:, n_firms - effective_k:]
        return np.sum(top_k, axis=1)
    elif ind == "IHH":
        return np.sum(shares_matrix ** 2, axis=1) * 10000.0
    elif ind == "ID":
        sq = shares_matrix ** 2
        sum_sq = np.sum(sq, axis=1, keepdims=True)
        h = sq / sum_sq
        return np.sum(h ** 2, axis=1)
    elif ind == "IE":
        safe_shares = np.where(shares_matrix > 0.0, shares_matrix, 1.0)
        log_shares = np.log(safe_shares)
        elementwise_entropy = np.where(shares_matrix > 0.0, shares_matrix * log_shares, 0.0)
        return -np.sum(elementwise_entropy, axis=1)
    raise ValueError(f"Indicador {indicator} no admitido.")

# ==============================================================================
# 3. INTERFAZ STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Simulador de Concentración de Mercado", layout="wide")

st.title("Organización Industrial: Análisis y Simulación de Concentración")
st.markdown(
    "Compara un escenario observado de cuotas de mercado frente a la distribución "
    "empírica estocástica obtenida mediante una simulación de Monte Carlo uniforme (Dirichlet $\\alpha = 1$)."
)

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    
    indicator = st.selectbox(
        "Indicador de Concentración",
        options=["CRk", "IHH", "ID", "IE"],
        index=1,
        help="CRk: Ratio k firmas | IHH: Herfindahl-Hirschman (0-10000) | ID: Dominancia | IE: Entropía de Theil"
    )

    n_firms = st.slider("Número de empresas ($N$)", min_value=2, max_value=100, value=6, step=1)

    k_val = 4
    if indicator == "CRk":
        max_k = max(1, n_firms - 1)
        k_val = st.number_input(
            f"Selecciona k (debe ser menor a N={n_firms})",
            min_value=1,
            max_value=max_k,
            value=min(4, max_k),
            step=1
        )

    n_iterations = st.number_input(
        "Iteraciones de Monte Carlo",
        min_value=100,
        max_value=10000,
        value=1000,
        step=500
    )

    if n_iterations > 3000:
        st.warning(
            f"Aviso de latencia: {n_iterations} iteraciones pueden demorar la reactividad "
            "y aumentar el consumo de memoria en la interfaz."
        )

# --- ÁREA PRINCIPAL: DEFINICIÓN DEL CASO PARTICULAR ---
st.subheader("1. Caso Particular de Mercado")

# Inicialización de estado para las cuotas del caso particular
if "shares_input" not in st.session_state or len(st.session_state["shares_input"]) != n_firms:
    # Generar un caso uniforme por defecto adaptado a N
    default_shares = np.ones(n_firms) / n_firms
    st.session_state["shares_input"] = ", ".join([f"{s * 100:.2f}" for s in default_shares])

col_btn, col_help = st.columns([1, 4])
with col_btn:
    if st.button("Generar aleatorio"):
        random_shares = np.random.default_rng().dirichlet(np.ones(n_firms))
        st.session_state["shares_input"] = ", ".join([f"{s * 100:.2f}" for s in random_shares])

raw_input = st.text_input(
    f"Cuotas individuales para N = {n_firms} empresas (separadas por comas, en % o decimales):",
    value=st.session_state["shares_input"]
)

# Parseo y procesamiento de cuotas ingresadas
shares_particular = None
parse_error = None

try:
    parsed_values = [float(x.strip()) for x in raw_input.split(",") if x.strip() != ""]
    if len(parsed_values) != n_firms:
        parse_error = f"Ingresaste {len(parsed_values)} cuotas, pero N está configurado en {n_firms}."
    else:
        # Validación con normalización forzada si hay residuo
        shares_particular = force_normalize_shares(parsed_values)
except Exception as e:
    parse_error = f"Error en la entrada: {str(e)}"

if parse_error:
    st.error(parse_error)
    st.stop()

# Cálculo del caso particular
particular_val = calculate_indicator(shares_particular, indicator=indicator, k=k_val)

# Mostrar métricas del caso particular
m1, m2, m3 = st.columns(3)
m1.metric("Valor del Caso Particular", f"{particular_val:.4f}")
m2.metric("Suma Total Cuotas", f"{np.sum(shares_particular) * 100:.1f}%")
m3.metric("Líder del Mercado", f"{np.max(shares_particular) * 100:.2f}%")

# --- SIMULACIÓN Y GRÁFICO ---
from scipy.stats import gaussian_kde

# --- SIMULACIÓN Y ANÁLISIS ESTADÍSTICO ---
st.subheader("2. Distribución Empírica de Monte Carlo vs. Caso Particular")

with st.spinner("Ejecutando simulación de Monte Carlo..."):
    sim_results = simulate_market_concentration_monte_carlo(
        n_firms=n_firms,
        n_iterations=int(n_iterations),
        indicator=indicator,
        k=k_val,
        random_state=42
    )

# 1. Cálculo del Percentil Exacto (Rango Percentil Empírico)
percentil_exacto = float(np.mean(sim_results <= particular_val) * 100.0)
mean_sim = float(np.mean(sim_results))
std_sim = float(np.std(sim_results))

# 2. Métricas en la interfaz de Streamlit
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric(
    label=f"Valor Caso ({indicator})",
    value=f"{particular_val:.4f}"
)
col_m2.metric(
    label="Percentil en Simulación",
    value=f"P{percentil_exacto:.1f}",
    help="Porcentaje de escenarios simulados con un nivel de concentración menor o igual al caso observado."
)
col_m3.metric(
    label="Media Monte Carlo (E[s])",
    value=f"{mean_sim:.4f}"
)
col_m4.metric(
    label="Desviación Estándar",
    value=f"{std_sim:.4f}"
)

# 3. Estimación de Densidad por Kernel (KDE)
kde = gaussian_kde(sim_results)
x_eval = np.linspace(sim_results.min(), sim_results.max(), 500)
kde_curve = kde(x_eval)

# 4. Construcción del Gráfico con Matplotlib
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)

# Histograma normalizado como densidad de probabilidad
n_counts, bin_edges, patches = ax.hist(
    sim_results,
    bins=35,
    density=True,
    alpha=0.45,
    color="#3b82f6",
    edgecolor="white",
    linewidth=0.8,
    label=f"Histograma de frecuencias (N={n_iterations})"
)

# Curva KDE superpuesta
ax.plot(
    x_eval,
    kde_curve,
    color="#1d4ed8",
    linewidth=2.2,
    label="Densidad Estimada (KDE Gaussiano)"
)

# Línea vertical del caso particular (roja, punteada y destacada)
ax.axvline(
    x=particular_val,
    color="#dc2626",
    linestyle="--",
    linewidth=2.5,
    zorder=5,
    label=f"Caso Particular: {particular_val:.4f} (P{percentil_exacto:.1f})"
)

# Línea de referencia de la media teórica/simulada
ax.axvline(
    x=mean_sim,
    color="#059669",
    linestyle=":",
    linewidth=1.8,
    alpha=0.9,
    label=f"Media simulada: {mean_sim:.4f}"
)

# Cuadro de anotación contextual sobre la línea del caso particular
y_max = ax.get_ylim()[1]
ax.annotate(
    f"Caso Particular\nValor: {particular_val:.4f}\nPercentil: {percentil_exacto:.1f}%",
    xy=(particular_val, y_max * 0.75),
    xytext=(15, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee2e2", edgecolor="#dc2626", alpha=0.95),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="#dc2626", lw=1.5),
    fontsize=9,
    fontweight="semibold",
    color="#7f1d1d"
)

# Configuración de etiquetas y unidades claras
unidades_x = {
    "CRk": f"Ratio de concentración acumulada CR_{k_val} (escala [0, 1])",
    "IHH": "Puntos de HHI (escala estándar [0, 10.000])",
    "ID": "Índice de Dominancia ID (escala [1/N, 1])",
    "IE": "Entropía de Theil (nats / escala continua)"
}

ax.set_title(
    f"Distribución Estocástica de Concentración vs. Caso Particular\n"
    f"Indicador: {indicator} | Empresas en Mercado: N = {n_firms} | Simulaciones: {n_iterations:,}",
    fontsize=12,
    fontweight="bold",
    pad=14
)
ax.set_xlabel(unidades_x.get(indicator, "Valor del Índice"), fontsize=10, labelpad=8)
ax.set_ylabel("Densidad de Probabilidad empírica [f(x)]", fontsize=10, labelpad=8)

# Estilo y legibilidad
ax.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="#cbd5e1", fontsize=9)

# Ajuste de márgenes
plt.tight_layout()

# Render en Streamlit
st.pyplot(fig)
# ==============================================================================
# 4. MÓDULO EVALUADOR Y RETROALIMENTACIÓN PEDAGÓGICA
# ==============================================================================
st.divider()
st.subheader("3. Autoevaluación y Diagnóstico de Concentración")

def get_theoretical_benchmark(ind: str, val: float, n: int, k: int) -> Tuple[str, str]:
    """
    Determina la clasificación económica teórica y la justificación normativa.
    
    Retorna:
    --------
    Tuple[str, str]:
        - Clasificación canónica: 'Desconcentrado / Competitivo', 
          'Moderadamente concentrado' o 'Altamente concentrado'.
        - Explicación de los umbrales normativos aplicados.
    """
    if ind == "IHH":
        if val < 1500.0:
            cat = "Desconcentrado / Competitivo"
            criterio = "IHH < 1.500 puntos (mercado competitivo con bajo poder de fijación unilateral)."
        elif 1500.0 <= val <= 2500.0:
            cat = "Moderadamente concentrado"
            criterio = "1.500 <= IHH <= 2.500 puntos (estructura oligopólica susceptible a escrutinio antitrust)."
        else:
            cat = "Altamente concentrado"
            criterio = "IHH > 2.500 puntos (alta concentración; riesgo significativo de coordinación tácita o dominancia)."
        return cat, criterio

    elif ind == "CRk":
        # Umbrales canónicos de Bain / Scherer & Ross (adaptados si k < 4)
        lim_inf = 0.40 * (k / 4.0) if k < 4 else 0.40
        lim_sup = 0.70 * (k / 4.0) if k < 4 else 0.70
        if val < lim_inf:
            cat = "Desconcentrado / Competitivo"
            criterio = f"CR_{k} < {lim_inf*100:.1f}% (las empresas líderes controlan una fracción reducida de la oferta)."
        elif lim_inf <= val <= lim_sup:
            cat = "Moderadamente concentrado"
            criterio = f"{lim_inf*100:.1f}% <= CR_{k} <= {lim_sup*100:.1f}% (oligopolio moderado con competencia entre líderes)."
        else:
            cat = "Altamente concentrado"
            criterio = f"CR_{k} > {lim_sup*100:.1f}% (las {k} firmas principales dominan la mayor parte del mercado)."
        return cat, criterio

    elif ind == "ID":
        # Umbrales empíricos de García Alba Idunate
        if val < 0.35:
            cat = "Desconcentrado / Competitivo"
            criterio = "ID < 0.35 (las participaciones cuadráticas están dispersas; no hay un líder asimétrico destacado)."
        elif 0.35 <= val <= 0.60:
            cat = "Moderadamente concentrado"
            criterio = "0.35 <= ID <= 0.60 (asimetría apreciable que otorga liderazgo relativo a la firma mayor)."
        else:
            cat = "Altamente concentrado"
            criterio = "ID > 0.60 (dominancia estructural clara por parte de la empresa de mayor tamaño)."
        return cat, criterio

    elif ind == "IE":
        # Entropía de Theil: el máximo teórico es ln(N) (máxima desconcentración/simetría)
        max_entropy = np.log(n)
        dist_to_max = max_entropy - val  # Pérdida de entropía por concentración

        if dist_to_max < 0.35:
            cat = "Desconcentrado / Competitivo"
            criterio = f"IE = {val:.3f} cercana a su cota máxima teórica ln({n}) = {max_entropy:.3f} (alta dispersión)."
        elif 0.35 <= dist_to_max <= 0.85:
            cat = "Moderadamente concentrado"
            criterio = f"IE = {val:.3f} refleja una pérdida moderada de información frente al óptimo simétrico ({max_entropy:.3f})."
        else:
            cat = "Altamente concentrado"
            criterio = f"IE = {val:.3f} significativamente por debajo del máximo simétrico ({max_entropy:.3f}), denotando fuerte asimetría."
        return cat, criterio

    raise ValueError(f"Indicador desconocido: {ind}")

# Obtener diagnóstico normativo correcto
correct_category, theoretical_rule = get_theoretical_benchmark(
    indicator, particular_val, n_firms, k_val
)

# Interfaz del usuario
st.markdown("**¿Cómo clasificarías el nivel de concentración de este caso particular según la teoría económica?**")

opciones_evaluacion = [
    "Desconcentrado / Competitivo",
    "Moderadamente concentrado",
    "Altamente concentrado"
]

user_selection = st.radio(
    label="Selecciona tu dictamen técnico:",
    options=opciones_evaluacion,
    index=None,
    key=f"eval_radio_{indicator}_{n_firms}_{particular_val:.4f}"
)

if st.button("Validar respuesta", type="primary"):
    if user_selection is None:
        st.info("Por favor, selecciona una categoría antes de validar.")
    else:
        is_correct = (user_selection == correct_category)
        
        if is_correct:
            st.success(f"**¡Correcto!** El mercado se clasifica como: **{correct_category}**.")
        else:
            st.error(f"**Incorrecto.** Clasificaste el mercado como *'{user_selection}'*, pero formalmente corresponde a: **{correct_category}**.")

        # Explicación económica y contraste con Monte Carlo
        with st.expander("Ver Dictamen Técnico y Justificación Económica", expanded=True):
            st.markdown(f"### Análisis Normativo ({indicator})")
            st.markdown(f"- **Valor Observado:** `{particular_val:.4f}`")
            st.markdown(f"- **Criterio Estándar:** {theoretical_rule}")
            
            st.markdown("### Posición Relativa en la Simulación Estocástica (Monte Carlo)")
            
            # Interpretación del percentil empírico
            # Nota: para Entropía (IE), mayor valor significa menor concentración
            if indicator == "IE":
                concentracion_relativa = 100.0 - percentil_exacto
                st.markdown(
                    f"El caso particular se ubica en el percentil **P{percentil_exacto:.1f}** de la distribución de entropía. "
                    f"Esto implica que en el **{concentracion_relativa:.1f}%** de los escenarios aleatorios con {n_firms} empresas, "
                    f"el mercado resultó ser *más concentrado* (menor entropía) que tu caso observado."
                )
            else:
                st.markdown(
                    f"El caso particular se ubica en el percentil **P{percentil_exacto:.1f}** de la distribución estocástica. "
                    f"Esto significa que el **{percentil_exacto:.1f}%** de las estructuras simuladas de forma puramente aleatoria (Dirichlet simétrica) "
                    f"presentan un nivel de concentración igual o inferior al registrado en tu muestra."
                )

            # Síntesis conceptual
            if percentil_exacto > 90.0 and indicator != "IE":
                st.warning(
                    "**Observación Estructural:** El percentil supera el 90%, lo que evidencia una asimetría atípica "
                    "difícilmente atribuible a variaciones aleatorias normales de cuotas; denota la presencia de firmas dominantes."
                )
            elif percentil_exacto < 10.0 and indicator != "IE":
                st.info(
                    "**Observación Estructural:** El percentil es excepcionalmente bajo (< 10%), lo que indica una "
                    "distribución de cuotas sustancialmente más homogénea o atomizada que el promedio estocástico."
                )
                