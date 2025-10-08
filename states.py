from typing import List, Tuple, Dict, Optional
import math

Space = Tuple[int, int, int, int]  # (x, y, w, h)
Rect  = Tuple[int, int]           # (w, h)


def _norm(v, den):
    """
    Normaliza valores entre -1 y 1 en lugar de 0 y 1

    Args:
        v: valor a normalizar
        den: denominador (valor máximo esperado)
    Returns:
        float: valor normalizado entre -1 y 1
    """
    if den == 0:
        return 0.0
    # Primero normalizamos entre 0 y 1
    norm_0_1 = float(v) / float(den)
    # Luego convertimos de [0,1] a [-1,1]
    return 2.0 * norm_0_1 - 1.0


def compute_area_rank(rect: Rect, all_rects: List[Rect]) -> float:
    """Calcula el ranking de área del rectángulo (1.0 = más grande, 0.0 = más pequeño)."""
    if not all_rects:
        return 0.0
    areas = sorted([r[0] * r[1] for r in all_rects], reverse=True)
    rect_area = rect[0] * rect[1]
    try:
        rank = areas.index(rect_area)
        return _norm(rank, len(areas) - 1) if len(areas) > 1 else 0.0
    except ValueError:
        return 0.0


def compute_size_category(area: float, max_area: float) -> float:
    """Categoriza el tamaño del rectángulo en 5 niveles."""
    if max_area == 0:
        return 0.0
    ratio = area / max_area
    if ratio < 0.1:
        return -1.0  # tiny
    elif ratio < 0.25:
        return -0.5  # small
    elif ratio < 0.5:
        return 0.0   # medium
    elif ratio < 0.75:
        return 0.5   # large
    else:
        return 1.0   # huge


def compute_fragmentation(spaces: List[Space], container_area: float) -> float:
    """Calcula fragmentación: ratio de espacios pequeños vs total de espacios."""
    if not spaces or container_area == 0:
        return 0.0
    small_threshold = container_area * 0.05  # espacios < 5% del contenedor
    small_count = sum(1 for s in spaces if (s[2] * s[3]) < small_threshold)
    return _norm(small_count, len(spaces)) if len(spaces) > 0 else 0.0


def _rect_features_optimized(rect: Rect, W: int, Href: int, all_rects: List[Rect]) -> List[float]:
    """Features OPTIMIZADAS de rectángulo (10 dimensiones).
    
    Solo características geométricas invariantes (no dependen del espacio).
    
    Returns:
        List[float]: Vector de 10 features
            0. h_n: altura normalizada
            1. w_n: ancho normalizado
            2. area_n: área normalizada
            3. aspect_ratio: h/w (qué tan alargado)
            4. perimeter_n: perímetro normalizado
            5. compactness: qué tan compacto es el rectángulo
            6. is_square: si es aproximadamente cuadrado
            7. diagonal_n: diagonal normalizada
            8. area_rank: ranking de área entre todos los rects
            9. size_category: categoría de tamaño discreta
    """
    w, h = rect
    area = w * h
    
    # Básicas normalizadas
    h_n = _norm(h, Href)
    w_n = _norm(w, W)
    area_n = _norm(area, W * Href)
    
    # Geométricas
    aspect_ratio = h / w if w > 0 else 0.0
    perimeter = 2 * (h + w)
    max_perimeter = 2 * (W + Href)
    perimeter_n = _norm(perimeter, max_perimeter)
    
    # Compactness: qué tan "compacto" es (cuadrado = alto, rect alargado = bajo)
    compactness = area / (w*w + h*h + 1e-6)
    
    # Es cuadrado? (tolerancia 10%)
    is_square = 1.0 if abs(h - w) / max(h, w) < 0.1 else -1.0
    
    # Diagonal
    diagonal = math.sqrt(h*h + w*w)
    max_diagonal = math.sqrt(W*W + Href*Href)
    diagonal_n = _norm(diagonal, max_diagonal)
    
    # Contexto relativo
    area_rank = compute_area_rank(rect, all_rects)
    size_category = compute_size_category(area, W * Href)
    
    return [
        h_n,           # 0
        w_n,           # 1
        area_n,        # 2
        aspect_ratio,  # 3
        perimeter_n,   # 4
        compactness,   # 5
        is_square,     # 6
        diagonal_n,    # 7
        area_rank,     # 8
        size_category, # 9
    ]


def _space_features_optimized(
    space: Space, 
    W: int, 
    Href: int, 
    all_spaces: List[Space],
    current_max_height: float = 0.0
) -> List[float]:
    """Features OPTIMIZADAS de espacio (12 dimensiones).
    
    Características geométricas y contexto espacial (sin features dependientes de rectángulos).
    
    Returns:
        List[float]: Vector de 12 features
            0. x_n: posición x normalizada
            1. y_n: posición y normalizada
            2. h_n: altura normalizada
            3. w_n: ancho normalizado
            4. area_n: área normalizada
            5. bottom_left_score: qué tan cerca de esquina inferior izquierda
            6. y_relative: altura relativa al máximo actual
            7. aspect_ratio: h/w del espacio
            8. is_tall: si es más alto que ancho
            9. utilization_potential: potencial de utilización
            10. num_spaces_n: cantidad de espacios normalizada
            11. fragmentation: nivel de fragmentación
    """
    x, y, w, h = space
    area = w * h
    
    # Básicas normalizadas
    x_n = _norm(x, W)
    y_n = _norm(y, Href)
    h_n = _norm(h, Href)
    w_n = _norm(w, W)
    area_n = _norm(area, W * Href)
    
    # Posición estratégica
    bottom_left_score = (1 - x/W) * (1 - y/Href) if W > 0 and Href > 0 else 0.0
    y_relative = y / max(current_max_height, 1.0) if current_max_height > 0 else 0.0
    
    # Forma del espacio
    aspect_ratio = h / w if w > 0 else 0.0
    is_tall = 1.0 if h > w else -1.0
    
    # Potencial de utilización
    utilization_potential = area / (W * Href) if (W * Href) > 0 else 0.0
    
    # Contexto del empaquetamiento
    num_spaces_n = _norm(len(all_spaces), 20)  # asumiendo max ~20 espacios
    fragmentation = compute_fragmentation(all_spaces, W * Href)
    
    return [
        x_n,                    # 0
        y_n,                    # 1
        h_n,                    # 2
        w_n,                    # 3
        area_n,                 # 4
        bottom_left_score,      # 5
        y_relative,             # 6
        aspect_ratio,           # 7
        is_tall,                # 8
        utilization_potential,  # 9
        num_spaces_n,           # 10
        fragmentation,          # 11
    ]

def codificar_estado(
    spaces: List[Space],
    rects: List[Rect],
    espacio_seleccionado: Space,  # S ACTIVO de este paso
    W: int, 
    Href: int,
    include_xy: bool = True,
) -> Dict[str, object]:
    """
    Devuelve SOLO tensores de entrada y máscaras (sin y_rect, sin seq_id).
    NOTA: Esta función se mantiene para compatibilidad con código legacy del transformer anterior.
    Para el modelo pointer, usa pointer_features() directamente.
    
    Returns:
        List con [S_in, R_in] donde:
        - S_in: lista de vectores de 12 features por espacio
        - R_in: lista de vectores de 10 features por rectángulo
    """
    # Calcular current_max_height para space features
    current_max_height = max([s[1] + s[3] for s in spaces]) if spaces else 0.0
    
    # Subespacios con nuevas features optimizadas
    S_in = []
    for s in spaces:
        S_in.append(_space_features_optimized(s, W, Href, spaces, current_max_height))

    # Rectángulos con nuevas features optimizadas
    R_in = []
    for r in rects:
        R_in.append(_rect_features_optimized(r, W, Href, rects))

    return [
        S_in,        # (K, 12)
        R_in         # (N, 10)
    ]


def pointer_features(
    spaces: List[Space],
    rects: List[Rect],
    active_space: Space,
    W: int,
    Href: int,
    include_xy: bool = True,
):
    """Construye features optimizados para el modelo pointer.

    Args:
        spaces: Lista de espacios disponibles
        rects: Lista de rectángulos restantes
        active_space: Espacio activo para este paso
        W: Ancho del contenedor
        Href: Altura de referencia
        include_xy: Ignorado (mantenido por compatibilidad, siempre incluye xy)

    Returns:
        space_feat: List[float]         -> (12,) features del espacio activo
        rect_feats: List[List[float]]   -> (N, 10) features de cada rectángulo
        feasible_mask: List[int]        -> (N,) 1 si cabe en active_space, 0 si no
    """
    # Calcular current_max_height
    current_max_height = max([s[1] + s[3] for s in spaces]) if spaces else 0.0
    
    # Feature del espacio activo (12 dims)
    space_feat = _space_features_optimized(active_space, W, Href, spaces, current_max_height)
    
    # Features de rectángulos (10 dims cada uno)
    rect_feats = []
    feasible_mask = []
    
    for r in rects:
        # Features invariantes del rectángulo
        feats = _rect_features_optimized(r, W, Href, rects)
        rect_feats.append(feats)
        
        # Calcular factibilidad respecto al espacio activo
        rw, rh = r
        _, _, sw, sh = active_space
        fits_normal = (rw <= sw and rh <= sh)
        fits_rotated = (rh <= sw and rw <= sh)
        feasible = fits_normal or fits_rotated
        feasible_mask.append(1 if feasible else 0)
    
    return space_feat, rect_feats, feasible_mask


def agregar_seq_id(samples):
    """
    Agrega seq_id al FINAL de cada vector en S_in y R_in, para todos los samples.
    seq_id = t / (T-1), donde t es el índice del paso, T = len(samples).
    Si T==1, seq_id=0.0.
    """
    T = len(samples)
    if T <= 1:
        seq_ids = [0.0] * T
    else:
        seq_ids = [t / (T - 1) for t in range(T)]

    for t, sample in enumerate(samples):
        sid = seq_ids[t]

        # Agregar seq_id al final de cada subespacio
        # sample["S_in"] tiene forma [S_in] -> (1, K, Fs); por eso iteramos sobre [0]
        for s in sample["S_in"][0]:
            s.append(sid)

        # Agregar seq_id al final de cada rectángulo
        for r in sample["R_in"][0]:
            r.append(sid)

    return samples


def agregar_seq_id_estados(all_states):
    """
    all_states: lista de pasos; cada paso es [S_in, R_in]
      - S_in: lista de vectores (subespacios)
      - R_in: lista de vectores (rectángulos), puede ser []
    Agrega/actualiza seq_id al FINAL de cada vector como t/(T-1).
    Devuelve una nueva lista (no muta la original).
    """
    import copy

    T = len(all_states)
    if T <= 1:
        seq_ids = [0.0] * T
    else:
        seq_ids = [t / (T - 1) for t in range(T)]

    out = copy.deepcopy(all_states)

    # Detectar longitud base de S y R (sin seq_id)
    base_len_S, base_len_R = None, None
    for step in out:
        # S_in
        if base_len_S is None and step and step[0]:
            for vec in step[0]:
                if isinstance(vec, (list, tuple)) and len(vec) > 0:
                    base_len_S = len(vec)
                    break
        # R_in
        if base_len_R is None and step and len(step) > 1 and step[1]:
            for vec in step[1]:
                if isinstance(vec, (list, tuple)) and len(vec) > 0:
                    base_len_R = len(vec)
                    break
        if base_len_S is not None and base_len_R is not None:
            break

    for t, step in enumerate(out):
        sid = seq_ids[t]

        # --- S_in ---
        if step and step[0]:
            for i, vec in enumerate(step[0]):
                if base_len_S is None:
                    # Si no se pudo inferir, intenta append directo
                    step[0][i] = list(vec) + [sid]
                else:
                    L = len(vec)
                    if L == base_len_S:
                        step[0][i] = list(vec) + [sid]
                    elif L == base_len_S + 1:
                        step[0][i][-1] = sid   # sobrescribe el seq_id existente
                    else:
                        # Longitud inesperada: aún así, agrega sid para no fallar
                        step[0][i] = list(vec) + [sid]

        # --- R_in ---
        if step and len(step) > 1 and step[1]:
            for i, vec in enumerate(step[1]):
                # Si nunca vimos R antes (todos vacíos hasta ahora), definimos base aquí
                if base_len_R is None:
                    base_len_R = len(vec)

                L = len(vec)
                if L == base_len_R:
                    step[1][i] = list(vec) + [sid]
                elif L == base_len_R + 1:
                    step[1][i][-1] = sid
                else:
                    step[1][i] = list(vec) + [sid]

    return out
