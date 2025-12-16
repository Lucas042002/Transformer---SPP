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


def _calcular_fit_quality(rect: Rect, space: Space) -> float:
    """Calcula qué tan bien encaja un rectángulo en un espacio.
    
    Args:
        rect: (w, h) rectángulo a evaluar
        space: (x, y, w, h) espacio disponible
        
    Returns:
        float: Score entre 0 (mal fit) y 1 (fit perfecto)
    """
    rect_w, rect_h = rect
    space_x, space_y, space_w, space_h = space
    
    if rect_w > space_w or rect_h > space_h:
        return 0.0
    
    # Ratio de área utilizada (más alto = mejor)
    rect_area = rect_w * rect_h
    space_area = space_w * space_h
    area_ratio = rect_area / space_area if space_area > 0 else 0.0
    
    # Similitud de aspect ratio (más cercano = mejor)
    rect_aspect = rect_w / rect_h if rect_h > 0 else 0.0
    space_aspect = space_w / space_h if space_h > 0 else 0.0
    aspect_diff = abs(rect_aspect - space_aspect)
    aspect_similarity = 1.0 / (1.0 + aspect_diff)  # Normalizar entre 0 y 1
    
    # Combinar métricas (70% área, 30% forma)
    fit_score = (area_ratio * 0.7) + (aspect_similarity * 0.3)
    
    return fit_score


def _calcular_compatibilidad_con_rectangulos(
    space: Space,
    remaining_rects: List[Rect],
    container_width: int,
    container_height: int,
    current_max_height: float = 0.0
) -> List[float]:
    """Calcula features de compatibilidad entre el espacio y los rectángulos disponibles.
    
    Args:
        space: (x, y, w, h) espacio a evaluar
        remaining_rects: Lista de rectángulos aún no colocados
        container_width: Ancho del contenedor
        container_height: Altura del contenedor
        current_max_height: Altura máxima actual del packing (para calcular impacto)
        
    Returns:
        List[float]: 8 features de compatibilidad:
            0. num_compatible_normalized: % de rectángulos que caben
            1. best_fit_ratio: % del espacio que usa el mejor candidato
            2. best_waste_normalized: desperdicio normalizado del mejor candidato
            3. best_aspect_match: similitud de forma con mejor candidato
            4. best_is_large: si el mejor candidato es grande (>mediana)
            5. avg_fit_quality: calidad promedio de fit de todos compatibles
            6. best_height_increase_normalized: cuánto subiría la altura con el mejor 
            7. best_promotes_horizontal: si el mejor promueve empaque horizontal
    """
    space_x, space_y, space_w, space_h = space
    space_area = space_w * space_h
    container_area = container_width * container_height
    
    if not remaining_rects or space_area == 0:
        # Sin rectángulos restantes: features por defecto
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Encontrar rectángulos compatibles (que caben)
    compatible = []
    fit_scores = []
    
    for rect in remaining_rects:
        rect_w, rect_h = rect
        if rect_w <= space_w and rect_h <= space_h:
            compatible.append(rect)
            fit_score = _calcular_fit_quality(rect, space)
            fit_scores.append(fit_score)
    
    # Si no hay compatibles, retornar features indicando "mal espacio"
    if not compatible:
        return [
            0.0,  # num_compatible_normalized: ninguno cabe
            0.0,  # best_fit_ratio: no hay candidato
            1.0,  # best_waste_normalized: máximo desperdicio
            0.0,  # best_aspect_match: no hay match
            0.0,  # best_is_large: no hay candidato
            0.0,  # avg_fit_quality: no hay calidad
            1.0,  # best_height_increase_normalized: sin candidato, penalizar
            0.0,  # best_promotes_horizontal: no hay candidato
        ]
    
    # Encontrar el mejor candidato
    best_idx = fit_scores.index(max(fit_scores))
    best_rect = compatible[best_idx]
    best_rect_w, best_rect_h = best_rect
    best_rect_area = best_rect_w * best_rect_h
    
    # Calcular features
    
    num_compatible_normalized = len(compatible) / len(remaining_rects)
    
    best_fit_ratio = best_rect_area / space_area
    
    best_waste = space_area - best_rect_area
    best_waste_normalized = best_waste / container_area
    
    rect_aspect = best_rect_w / best_rect_h if best_rect_h > 0 else 0.0
    space_aspect = space_w / space_h if space_h > 0 else 0.0
    aspect_diff = abs(rect_aspect - space_aspect)
    best_aspect_match = 1.0 / (1.0 + aspect_diff)
    
    all_areas = [r[0] * r[1] for r in remaining_rects]
    median_area = sorted(all_areas)[len(all_areas) // 2]
    best_is_large = 1.0 if best_rect_area > median_area else 0.0
    
    avg_fit_quality = sum(fit_scores) / len(fit_scores)
    
    # Si el espacio está arriba de current_max_height, colocar el rect INCREMENTA la altura
    space_top = space_y + space_h  # Límite superior del espacio
    new_height_with_best = space_y + best_rect_h  # Nueva altura si colocamos el mejor
    
    if new_height_with_best > current_max_height:
        # Este espacio INCREMENTARÍA la altura
        height_increase = new_height_with_best - current_max_height
        best_height_increase_normalized = _norm(height_increase, container_height)
    else:
        # Este espacio NO incrementa altura (está en niveles ya ocupados)
        best_height_increase_normalized = -1.0  # Señal de "no incrementa"
    
    # Rectángulos más anchos que altos son mejores para minimizar altura
    best_promotes_horizontal = 1.0 if best_rect_w > best_rect_h else -1.0
    
    return [
        num_compatible_normalized,
        best_fit_ratio,
        best_waste_normalized,
        best_aspect_match,
        best_is_large,
        avg_fit_quality,
        best_height_increase_normalized,  
        best_promotes_horizontal,         
    ]


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
    
    h_n = _norm(h, Href)
    w_n = _norm(w, W)
    area_n = _norm(area, W * Href)
    
    aspect_ratio = h / w if w > 0 else 0.0
    perimeter = 2 * (h + w)
    max_perimeter = 2 * (W + Href)
    perimeter_n = _norm(perimeter, max_perimeter)
    
    compactness = area / (w*w + h*h + 1e-6)
    
    is_square = 1.0 if abs(h - w) / max(h, w) < 0.1 else -1.0
    
    diagonal = math.sqrt(h*h + w*w)
    max_diagonal = math.sqrt(W*W + Href*Href)
    diagonal_n = _norm(diagonal, max_diagonal)
    
    area_rank = compute_area_rank(rect, all_rects)
    size_category = compute_size_category(area, W * Href)
    
    return [
        h_n,           
        w_n,           
        area_n,        
        aspect_ratio,  
        perimeter_n,   
        compactness,   
        is_square,     
        diagonal_n,    
        area_rank,     
        size_category, 
    ]


def _space_features_optimized(
    space: Space, 
    W: int, 
    Href: int, 
    all_spaces: List[Space],
    current_max_height: float = 0.0,
    remaining_rects: Optional[List[Rect]] = None
) -> List[float]:
    """Features OPTIMIZADAS de espacio (17 dimensiones).
    
    Características geométricas, contexto espacial y compatibilidad con rectángulos disponibles.
    
    Args:
        space: (x, y, w, h) espacio a evaluar
        W: Ancho del contenedor
        Href: Altura de referencia
        all_spaces: Lista de todos los espacios (para contexto)
        current_max_height: Altura máxima actual del packing
        remaining_rects: Lista de rectángulos aún disponibles (para features de compatibilidad)
    
    Returns:
        List[float]: Vector de 19 features
            BÁSICAS (10):
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
            
            CONTEXTO GLOBAL (1):
            10. fragmentation: nivel de fragmentación
            
            COMPATIBILIDAD CON RECTÁNGULOS (8):
            11. num_compatible_normalized: % de rectángulos que caben
            12. best_fit_ratio: % del espacio que usa el mejor candidato
            13. best_waste_normalized: desperdicio normalizado del mejor
            14. best_aspect_match: similitud de forma con mejor candidato
            15. best_is_large: si el mejor candidato es grande
            16. avg_fit_quality: calidad promedio de fit
            17. best_height_increase_normalized: incremento de altura 
            18. best_promotes_horizontal: si promueve empaque horizontal 
    """
    x, y, w, h = space
    area = w * h
    
    # ========== FEATURES BÁSICAS (10) ==========
    x_n = _norm(x, W)
    y_n = _norm(y, Href)
    h_n = _norm(h, Href)
    w_n = _norm(w, W)
    area_n = _norm(area, W * Href)
    
    bottom_left_score = (1 - x/W) * (1 - y/Href) if W > 0 and Href > 0 else 0.0
    y_relative = y / max(current_max_height, 1.0) if current_max_height > 0 else 0.0
    
    aspect_ratio = h / w if w > 0 else 0.0
    is_tall = 1.0 if h > w else -1.0
    
    utilization_potential = area / (W * Href) if (W * Href) > 0 else 0.0
    
    basic_features = [
        x_n,                    
        y_n,                    
        h_n,                    
        w_n,                    
        area_n,                 
        bottom_left_score,      
        y_relative,             
        aspect_ratio,           
        is_tall,                
        utilization_potential,  
    ]
    
    # ========== CONTEXTO GLOBAL (1) ==========
    fragmentation = compute_fragmentation(all_spaces, W * Href)
    
    # ========== FEATURES DE COMPATIBILIDAD (8) ==========
    if remaining_rects is not None and len(remaining_rects) > 0:
        compatibility_features = _calcular_compatibilidad_con_rectangulos(
            space,
            remaining_rects,
            W,
            Href,
            current_max_height 
        )
    else:
        # Sin rectángulos: features por defecto (ahora 8)
        # [num_compatible, best_fit_ratio, best_waste(1.0=máximo), best_aspect, best_is_large, avg_fit, height_increase, promotes_horizontal]
        compatibility_features = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return basic_features + [fragmentation] + compatibility_features

def codificar_estado(
    spaces: List[Space],
    rects: List[Rect],
    espacio_seleccionado: Space, 
    W: int, 
    Href: int,
    include_xy: bool = True,
) -> Dict[str, object]:
    """
    Devuelve SOLO tensores de entrada y máscaras (sin y_rect, sin seq_id).
    Para el modelo pointer, usa pointer_features() directamente.
    
    Returns:
        List con [S_in, R_in] donde:
        - S_in: lista de vectores de 19 features por espacio
        - R_in: lista de vectores de 10 features por rectángulo
    """
    # Calcular current_max_height para space features
    current_max_height = max([s[1] + s[3] for s in spaces]) if spaces else 0.0
    
    # Subespacios con nuevas features optimizadas
    S_in = []
    for s in spaces:
        S_in.append(_space_features_optimized(s, W, Href, spaces, current_max_height, remaining_rects=None))

    # Rectángulos con nuevas features optimizadas
    R_in = []
    for r in rects:
        R_in.append(_rect_features_optimized(r, W, Href, rects))

    return [
        S_in,        # (K, 19)
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
        space_feat: List[float]         -> (19,) features del espacio activo
        rect_feats: List[List[float]]   -> (N, 10) features de cada rectángulo
        feasible_mask: List[int]        -> (N,) 1 si cabe en active_space, 0 si no
    """
    # Calcular current_max_height
    current_max_height = max([s[1] + s[3] for s in spaces]) if spaces else 0.0
    
    # Feature del espacio activo (19 dims)
    space_feat = _space_features_optimized(
        active_space, 
        W, 
        Href, 
        spaces, 
        current_max_height,
        remaining_rects=rects 
    )
    
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

