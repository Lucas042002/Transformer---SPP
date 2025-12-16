import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List, Tuple
from dataclasses import dataclass

import torch

import states as st
import categories as cat

# Type aliases
Rect = Tuple[int, int]
Space = Tuple[int, int, int, int]

# --------------------------------------------------
# PointerStep dataclass (para entrenamiento pointer)
# --------------------------------------------------
@dataclass
class PointerStep:
    """Representa un paso de decisión en la trayectoria de empaquetamiento.
    
    Attributes:
        rect_feats: (N, 10) features de N rectángulos ORIGINALES (tamaño FIJO durante toda la trayectoria)
        rect_mask: (N,) bool - True si el rectángulo está disponible Y es factible en el espacio actual
        space_feat: (12,) features del espacio activo
        target: índice del rectángulo elegido en el conjunto ORIGINAL (0..N-1)
        step_idx: número de paso en la secuencia (para step embedding)
    
    IMPORTANTE: rect_feats tiene SIEMPRE tamaño (N, 10) donde N es el número de rectángulos
    originales del problema. La máscara rect_mask indica cuáles están disponibles (no colocados)
    y son factibles (caben) en el espacio actual.
    """
    rect_feats: torch.Tensor  # (N_original, 10) - TAMAÑO FIJO
    rect_mask: torch.Tensor   # (N_original,) bool (disponible Y factible)
    space_feat: torch.Tensor  # (12,)
    target: int               # índice en conjunto ORIGINAL (0..N_original-1)
    step_idx: int

# ----------------------------
# Funciones del algoritmo HR (simplificado)
# ----------------------------
def rect_fits_in_space(rect, space):
    rw, rh = rect
    x, y, w, h = space

    # Prueba sin rotar y rotado
    # Sin rotar
    if rw <= w and rh <= h:
        result = True, 0
        return result
    # Rotado
    if rh <= w and rw <= h:
        result = True, 1
        return result
    result = False, -1
    return result

def place_rect(space, rect):
    if rect_fits_in_space(rect, space):
        return True, (space[0], space[1])  
    return False, (-1, -1)

def divide_space(space, rect, pos):
    x, y, w, h = space        # espacio original: (x, y, ancho, alto)
    rw, rh = rect             # dimensiones del rectángulo insertado
    rx, ry = pos              # posición donde fue colocado

    # S1: espacio encima del rectángulo (horizontal, unbounded en alto)
    S1 = (x, ry + rh, w, y + h - (ry + rh))

    # S2: espacio a la derecha (bounded por altura del rectángulo)
    S2 = (rx + rw, y, x + w - (rx + rw), rh)

    return S1, S2


# Dividir el espacio en S3 y S4
def divide_space_2(space, rect, pos):
    x, y, w, h = space        # espacio original: (x, y, ancho, alto)
    rw, rh = rect             # dimensiones del rectángulo insertado
    rx, ry = pos              # posición donde fue colocado

    # S3: espacio arriba del rectángulo (bounded por altura del S2)
    S3 = (rx + rw, y, x + w - (rx + rw), h)

    # S4: espacio a la derecha (bounded por altura del S2)
    S4 = (x, ry + rh, rw, y + h - (ry + rh))

    return S3, S4


def codificar_y_rect_con_lista(rects_pendientes, rect):
    """
    rects_pendientes: lista [(w,h), ...] en el orden en que armaste R_in[0]
    rect: (w,h) elegido por HR
    Devuelve el índice entero (int) dentro de rects_pendientes donde se encuentra
    el rectángulo `rect` seleccionado por el HR en este paso.
    Devuelve el índice (1-based) o 0 si no se encuentra.
    """
    N = len(rects_pendientes)
    Y = 0

    for idx, r in enumerate(rects_pendientes):
        if r == rect or r == (rect[1], rect[0]):
            Y = idx+1
            break

    return Y


def recursive_packing(space, spaces, rects, placed, Y_rect=None, category=""):
    while rects:
        for i, rect in enumerate(rects):
            # Verificar si el rectángulo cabe en el espacio
            fits, rotation = rect_fits_in_space(rect, space)
            if fits:
                # Rotar el rectángulo si es necesario
                if rotation == 1:
                    rect = (rect[1], rect[0])

                ok, pos = place_rect(space, rect)
                if ok:
                    placed.append((rect, pos))
                    Y_rect.append(codificar_y_rect_con_lista(rects, rect))

                    # Dividir en nuevos S3 y S4 desde el subespacio actual
                    S3, S4 = divide_space_2(space, rect, pos)
                    temp_spaces = spaces.copy()
                    temp_spaces.append(S3)
                    temp_spaces.append(S4)

                    # Eliminar rectángulo usado
                    rects.pop(i)
                    area_S3 = S3[2] * S3[3]
                    area_S4 = S4[2] * S4[3]
                    
                    if not rects:
                        return

                    if area_S3 > area_S4:
                        recursive_packing(S3, spaces, rects, placed, Y_rect, category)
                        temp_spaces.remove(S3)

                        if not rects:
                            return

                        recursive_packing(S4, spaces, rects, placed, Y_rect, category)
                        temp_spaces.remove(S4)
                    else:
                        recursive_packing(S4, spaces, rects, placed, Y_rect, category)
                        temp_spaces.remove(S4)
                        
                        if not rects:
                            return

                        recursive_packing(S3, spaces, rects, placed, Y_rect, category)
                        temp_spaces.remove(S3)

                    return
        break

def hr_packing(spaces, rects, category=""):
    placed = []
    rects1 = rects.copy()
    Y_rect = []

    while rects1:
        placed_flag = False
        for space in spaces:
            for i, rect in enumerate(rects1):
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    if ok:
                        temp_spaces = []
                        placed.append((rect, pos))
                        Y_rect.append(codificar_y_rect_con_lista(rects1, rect))

                        # Dividir el espacio en S1 (encima, unbounded) y S2 (derecha, bounded)
                        S1, S2 = divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)

                        # Eliminar rectángulo insertado y espacio usado
                        rects1.pop(i)
                        spaces.remove(space)
                            
                        # Agregar S1 al espacio disponible para seguir iterando
                        spaces.append(S1)
                        # Llamar recursivamente a RecursivePacking con S2 (bounded)
                        recursive_packing(S2, spaces, rects1, placed, Y_rect, category)

                        placed_flag = True
                        break  
            if placed_flag:
                break
        if not placed_flag:
            break
    
    return placed, Y_rect


# --------------------------------------------------
# Versión optimizada para entrenamiento del Pointer Model
# --------------------------------------------------
def recursive_packing_pointer(
    space: Space, 
    spaces: List[Space], 
    rects: List[Rect], 
    placed: List[Tuple[Rect, Tuple[int, int]]], 
    pointer_steps: List[PointerStep],
    container_width: int,
    Href: int,
    step_counter: List[int],  # Contador mutable para step_idx
    rects_originales: List[Rect],  # Lista original completa
    all_rect_feats: List[List[float]],  # Features pre-calculadas
    used_indices: set,  # Conjunto de índices ya usados (para manejar duplicados)
    Y_rect: List[int] = None  # Lista de índices elegidos (1-based)
):
    """Versión de recursive_packing que genera PointerSteps con features optimizados.
    
    IMPORTANTE: Usa features pre-calculadas de rects_originales (calculadas UNA sola vez)
    para garantizar consistencia con inferencia.
    """
    if Y_rect is None:
        Y_rect = []
    while rects:
        for i, rect in enumerate(rects):
            fits, rotation = rect_fits_in_space(rect, space)
            if fits:
                if rotation == 1:
                    rect = (rect[1], rect[0])
                
                ok, pos = place_rect(space, rect)
                if ok:
                    # === GENERAR POINTERSTEP ANTES DE MODIFICAR EL ESTADO ===
                    current_rects = rects.copy()
                    current_spaces = spaces + [space]  # Incluir espacio activo
                    
                    # Mapear rectángulos restantes a sus índices originales
                    remaining_indices = []
                    for r in current_rects:
                        # Buscar el siguiente índice NO usado de este rectángulo (considerar rotación)
                        for idx, orig_rect in enumerate(rects_originales):
                            # Comparar considerando rotación: (w,h) == (w,h) o (w,h) == (h,w)
                            matches = (orig_rect == r) or (orig_rect == (r[1], r[0]))
                            if matches and idx not in used_indices:
                                remaining_indices.append(idx)
                                break
                    
                    # USAR TODAS LAS FEATURES (tamaño fijo = N_original)
                    # NO filtrar, mantener todas las features calculadas
                    rect_feats_tensor = torch.tensor(all_rect_feats, dtype=torch.float32)  # (N_original, 10)
                    
                    # Calcular space features (SÍ dependen del estado actual)
                    space_feat_vec, _, _ = st.pointer_features(
                        spaces=current_spaces,
                        rects=current_rects,
                        active_space=space,
                        W=container_width,
                        Href=Href
                    )
                    
                    # MÁSCARA: indica disponibilidad Y factibilidad
                    # Primero marcar cuáles están disponibles (no han sido colocados)
                    rect_mask_list = [False] * len(rects_originales)  # Inicialmente todos NO disponibles
                    
                    # Marcar los que SÍ están disponibles y verificar factibilidad
                    for idx in remaining_indices:
                        r = rects_originales[idx]
                        rw, rh = r
                        _, _, sw, sh = space
                        fits_normal = (rw <= sw and rh <= sh)
                        fits_rotated = (rh <= sw and rw <= sh)
                        feasible = fits_normal or fits_rotated
                        rect_mask_list[idx] = feasible  # True si disponible Y factible
                    
                    # Target: buscar el índice ORIGINAL del rectángulo elegido (NO usado)
                    # Considerar rotación: el rectángulo puede estar rotado
                    target_original_idx = None
                    for idx, orig_rect in enumerate(rects_originales):
                        matches = (orig_rect == rect) or (orig_rect == (rect[1], rect[0]))
                        if matches and idx not in used_indices:
                            target_original_idx = idx
                            break
                    
                    # Validar que el target sea factible
                    if target_original_idx is not None and rect_mask_list[target_original_idx]:
                        pointer_steps.append(PointerStep(
                            rect_feats=rect_feats_tensor,  # (N_original, 10) - tamaño FIJO
                            rect_mask=torch.tensor(rect_mask_list, dtype=torch.bool),  # (N_original,)
                            space_feat=torch.tensor(space_feat_vec, dtype=torch.float32),
                            target=target_original_idx,  # Índice en el conjunto ORIGINAL
                            step_idx=step_counter[0],
                        ))
                        step_counter[0] += 1
                        # Marcar este índice como usado
                        used_indices.add(target_original_idx)
                        # Guardar índice elegido (1-based para compatibilidad)
                        Y_rect.append(i + 1)
                    
                    # === CONTINUAR CON LA LÓGICA NORMAL ===
                    placed.append((rect, pos))
                    S3, S4 = divide_space_2(space, rect, pos)
                    temp_spaces = spaces.copy()
                    temp_spaces.append(S3)
                    temp_spaces.append(S4)
                    
                    rects.pop(i)
                    area_S3 = S3[2] * S3[3]
                    area_S4 = S4[2] * S4[3]
                    
                    if not rects:
                        return
                    
                    if area_S3 > area_S4:
                        recursive_packing_pointer(S3, spaces, rects, placed, pointer_steps, container_width, Href, step_counter, rects_originales, all_rect_feats, used_indices, Y_rect)
                        temp_spaces.remove(S3)
                        
                        if not rects:
                            return
                        
                        recursive_packing_pointer(S4, spaces, rects, placed, pointer_steps, container_width, Href, step_counter, rects_originales, all_rect_feats, used_indices, Y_rect)
                        temp_spaces.remove(S4)
                    else:
                        recursive_packing_pointer(S4, spaces, rects, placed, pointer_steps, container_width, Href, step_counter, rects_originales, all_rect_feats, used_indices, Y_rect)
                        temp_spaces.remove(S4)
                        
                        if not rects:
                            return
                        
                        recursive_packing_pointer(S3, spaces, rects, placed, pointer_steps, container_width, Href, step_counter, rects_originales, all_rect_feats, used_indices, Y_rect)
                        temp_spaces.remove(S3)
                    
                    return
        break


def hr_packing_pointer(spaces: List[Space], rects: List[Rect], category: str) -> Tuple[List[PointerStep], List[Tuple[Rect, Tuple[int, int]]], List[int]]:
    """Ejecuta el algoritmo HR y devuelve PointerSteps con features optimizados (10 dims rect, 12 dims space).
    
    Esta es la función de interfaz limpia para entrenamiento del Pointer Model.
    Genera directamente los PointerSteps sin necesidad de re-simular el proceso.
    
    IMPORTANTE: Las features de rectángulos se calculan UNA SOLA VEZ usando TODOS los rectángulos
    originales (como en inferencia), no se recalculan con los rectángulos restantes.
    Esto garantiza consistencia entre entrenamiento e inferencia.
    
    Args:
        spaces: Lista de espacios iniciales (normalmente [(0, 0, W, 1000)])
        rects: Lista de rectángulos a empaquetar
        category: Categoría del problema (determina W y Href)
        
    Returns:
        Tuple containing:
            - pointer_steps: Lista de PointerStep con features optimizados para entrenamiento
            - placed: Lista de (rect, position) con los rectángulos colocados
            - Y_rect: Lista de índices elegidos (1-based, compatible con formato antiguo para debugging)
    """
    placed = []
    rects1 = rects.copy()
    spaces = spaces.copy()
    pointer_steps: List[PointerStep] = []
    Y_rect: List[int] = []  # Lista de índices elegidos (1-based para debugging)
    step_counter = [0]  # Lista mutable para compartir entre llamadas recursivas
    
    W = cat.CATEGORIES[category]["width"]
    Href = cat.CATEGORIES[category]["height"]
    
    # OPTIMIZACIÓN: Calcular features de TODOS los rectángulos UNA SOLA VEZ
    # (igual que en inferencia con cached encoder)
    rects_originales = rects.copy()
    all_rect_feats = []
    for r in rects_originales:
        feats = st._rect_features_optimized(r, W, Href, rects_originales)
        all_rect_feats.append(feats)
    
    # Rastrear qué índices ya fueron usados (para manejar rectángulos duplicados)
    used_indices = set()
    
    # Main loop
    while rects1:
        placed_flag = False
        
        for space in spaces:
            for i, rect in enumerate(rects1):
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    if ok:
                        # === GENERAR POINTERSTEP ===
                        # Mapear rectángulos restantes a sus índices originales
                        # IMPORTANTE: Manejar duplicados correctamente y considerar rotación
                        remaining_indices = []
                        for r in rects1:
                            # Buscar el siguiente índice NO usado de este rectángulo (considerar rotación)
                            for idx, orig_rect in enumerate(rects_originales):
                                matches = (orig_rect == r) or (orig_rect == (r[1], r[0]))
                                if matches and idx not in used_indices:
                                    remaining_indices.append(idx)
                                    break
                        
                        # USAR TODAS LAS FEATURES (tamaño fijo = N_original)
                        rect_feats_tensor = torch.tensor(all_rect_feats, dtype=torch.float32)  # (N_original, 10)
                        
                        # Calcular space features (SÍ dependen del estado actual)
                        space_feat_vec, _, _ = st.pointer_features(
                            spaces=spaces,
                            rects=rects1,
                            active_space=space,
                            W=W,
                            Href=Href
                        )
                        
                        # MÁSCARA: indica disponibilidad Y factibilidad
                        rect_mask_list = [False] * len(rects_originales)
                        
                        # Marcar los que SÍ están disponibles y verificar factibilidad
                        for idx in remaining_indices:
                            r = rects_originales[idx]
                            rw, rh = r
                            _, _, sw, sh = space
                            fits_normal = (rw <= sw and rh <= sh)
                            fits_rotated = (rh <= sw and rw <= sh)
                            feasible = fits_normal or fits_rotated
                            rect_mask_list[idx] = feasible
                        
                        # Target: buscar el índice ORIGINAL del rectángulo elegido (NO usado)
                        # Considerar rotación: el rectángulo puede estar rotado
                        target_original_idx = None
                        for idx, orig_rect in enumerate(rects_originales):
                            matches = (orig_rect == rect) or (orig_rect == (rect[1], rect[0]))
                            if matches and idx not in used_indices:
                                target_original_idx = idx
                                break
                        
                        if target_original_idx is not None and rect_mask_list[target_original_idx]:
                            pointer_steps.append(PointerStep(
                                rect_feats=rect_feats_tensor,  # (N_original, 10) - tamaño FIJO
                                rect_mask=torch.tensor(rect_mask_list, dtype=torch.bool),  # (N_original,)
                                space_feat=torch.tensor(space_feat_vec, dtype=torch.float32),
                                target=target_original_idx,  # Índice en el conjunto ORIGINAL
                                step_idx=step_counter[0],
                            ))
                            step_counter[0] += 1
                            # Marcar este índice como usado
                            used_indices.add(target_original_idx)
                            # Guardar índice elegido (1-based para compatibilidad)
                            Y_rect.append(i + 1)
                        
                        # === LÓGICA NORMAL ===
                        temp_spaces = []
                        placed.append((rect, pos))
                        
                        S1, S2 = divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)
                        
                        rects1.pop(i)
                        spaces.remove(space)
                        
                        spaces.append(S1)
                        recursive_packing_pointer(S2, spaces, rects1, placed, pointer_steps, W, Href, step_counter, rects_originales, all_rect_feats, used_indices, Y_rect)
                        
                        placed_flag = True
                        break
            if placed_flag:
                break
        if not placed_flag:
            break
    
    return pointer_steps, placed, Y_rect 


def heuristic_recursion_pointer(rects: List[Rect], container_width: int, category: str) -> Tuple[List[PointerStep], List[Tuple[Rect, Tuple[int, int]]], int, List[Rect], List[int]]:
    """Versión optimizada de heuristic_recursion que devuelve PointerSteps.
    
    Prueba permutaciones de rectángulos ordenados por área para encontrar la mejor solución.
    Devuelve directamente los PointerSteps sin necesidad de re-simulación.
    
    Args:
        rects: Lista de rectángulos a empaquetar
        container_width: Ancho del contenedor
        category: Categoría del problema (determina W y Href)
        
    Returns:
        Tuple containing:
            - best_pointer_steps: Lista de PointerStep de la mejor solución
            - best_placements: Lista de (rect, position) de la mejor solución
            - best_height: Altura alcanzada en la mejor solución
            - best_rect_sequence: Secuencia de rectángulos que dio la mejor solución
            - best_Y_rect: Lista de índices elegidos (1-based) de la mejor solución
    """
    rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    best_rect_sequence = []
    best_pointer_steps = []
    best_Y_rect = []

    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            
            # Usar hr_packing_pointer en lugar de hr_packing
            pointer_steps, placements, Y_rect = hr_packing_pointer(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects,
                category=category
            )
            
            # Calcular altura
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0

            if altura < best_height:
                best_rect_sequence = temp_rects.copy()
                best_height = altura
                best_placements = placements
                best_pointer_steps = pointer_steps
                best_Y_rect = Y_rect

    return best_pointer_steps, best_placements, best_height, best_rect_sequence, best_Y_rect


# ----------------------------
def ordenar_por_area(rects):
    return sorted(rects, key=lambda r: r[0] * r[1], reverse=True)

def calcular_altura(placements):
    return max([y + h for (_, (x, y)), (w, h) in zip(placements, [p[0] for p in placements])], default=0)

def heuristic_recursion(rects, container_width, category=""):
    rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    rect_sequence = []
    best_Y_rect = []

    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            
            placements, Y_rect = hr_packing(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects,
                category=category
            )
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0

            if altura < best_height:
                rect_sequence = temp_rects.copy()
                best_height = altura
                best_placements = placements
                best_Y_rect = Y_rect

    return best_placements, best_height, rect_sequence, best_Y_rect
# ----------------------------
# Funciones de visualización
def visualizar_packing(placements, container_width, container_height=None, show=True):
    fig, ax = plt.subplots()
    colors = {}

    used_heights = [pos[1] + rect[1] for rect, pos in placements]
    max_height = max(used_heights) if used_heights else 0
    
    # Dibujar contenedor
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, max_height)
    ax.set_aspect('equal')
    ax.set_title("Visualización del HR Packing")
    ax.set_xlabel("Ancho")
    ax.set_ylabel("Altura")

    # Ejes con números enteros
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for i, (rect, pos) in enumerate(placements):
        w, h = rect
        x, y = pos
        color = colors.get(rect)
        if color is None:
            color = [random.random() for _ in range(3)]
            colors[rect] = color
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color, label=f"{rect}"))
        ax.text(x + w/2, y + h/2, f"{i}", fontsize=8, ha='center', va='center', color='black')

    # Línea roja en la altura del contenedor
    ax.axhline(y=container_height, color='red', linestyle='--', linewidth=2, label='Altura contenedor')

    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    return fig
