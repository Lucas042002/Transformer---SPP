"""Heurística de packing usando el modelo pointer (SPPPointerModel).

Sigue EXACTAMENTE la misma estructura que hr_algorithm.py pero usa el modelo pointer
para tomar las decisiones en lugar del algoritmo HR puro.

Mantiene las mismas funciones conocidas:
- rect_fits_in_space, place_rect, divide_space, divide_space_2
- recursive_packing_pointer (equivale a recursive_packing)  
- hr_packing_pointer (equivale a hr_packing)
- heuristic_recursion_pointer (equivale a heuristic_recursion)
- visualizar_packing (usa la misma función de hr_algorithm)
"""
from typing import List, Tuple
import torch

import categories as cat
import states as st
from pointer_model import SPPPointerModel
import hr_algorithm as hr  # Para reutilizar visualizar_packing y funciones geométricas

Space = Tuple[int, int, int, int]  # (x, y, w, h)
Rect = Tuple[int, int]


# --------------------------------------------------
# Funciones geométricas (reutilizamos las de hr_algorithm)
# --------------------------------------------------
def rect_fits_in_space(rect, space):
    return hr.rect_fits_in_space(rect, space)

def place_rect(space, rect):
    if rect_fits_in_space(rect, space):
        return True, (space[0], space[1])  # bottom-left
    return False, (-1, -1)

def divide_space(space, rect, pos):
    return hr.divide_space(space, rect, pos)

def divide_space_2(space, rect, pos):
    return hr.divide_space_2(space, rect, pos)

def codificar_y_rect_con_lista_pointer(rects_pendientes, rect_elegido_idx):
    """
    Convierte el índice dinámico (posición en lista remaining) a índice 1-based
    para mantener compatibilidad con la estructura de Y del HR original.
    """
    return rect_elegido_idx + 1  # Convertir 0-based a 1-based


def inicializar_cache_encoder(rects_originales, spaces, container_width, category, model, device):
    """
    Función auxiliar para inicializar el cache del encoder UNA SOLA VEZ.
    Calcula embeddings de todos los rectángulos originales para reutilizar.
    
    Returns:
        cached_rect_enc: embeddings pre-calculados (1, N, d_model)
        cached_global_ctx: NO SE USA (decode_step lo recalcula dinámicamente)
    """
    Href = cat.CATEGORIES[category]["height"]
    
    # Construir features de TODOS los rectángulos originales (10 dims cada uno)
    rect_feats_list = []
    for r in rects_originales:
        feats = st._rect_features_optimized(r, container_width, Href, rects_originales)
        rect_feats_list.append(feats)
    
    # Encodear UNA SOLA VEZ todos los rectángulos originales
    all_rect_feats = torch.tensor(rect_feats_list, dtype=torch.float32, device=device).unsqueeze(0)
    all_rect_mask = torch.ones(1, len(rects_originales), dtype=torch.bool, device=device)
    cached_rect_enc, _ = model.encode_rects(all_rect_feats, all_rect_mask)
    
    return cached_rect_enc, None  # global_ctx no se cachea (se recalcula dinámicamente)


def usar_modelo_pointer_para_decision_optimized(
    remaining_rects, 
    active_space, 
    model, 
    container_width, 
    category, 
    device, 
    step,
    rects_originales,
    cached_rect_enc, 
    cached_global_ctx  # Ya no se usa, decode_step lo recalcula
):
    """
    Versión OPTIMIZADA que reutiliza embeddings del encoder pre-calculados.
    Solo ejecuta el decoder en cada paso.
    
    El global_ctx se recalcula dinámicamente dentro de decode_step basado
    en la máscara de factibilidad actual.
    
    Returns:
        rect_idx: índice en remaining_rects del rectángulo elegido
        rect: el rectángulo (w,h) elegido
    """
    if not remaining_rects:
        return -1, None
        
    Href = cat.CATEGORIES[category]["height"]
    
    # Mapear remaining_rects a índices en la lista original
    remaining_indices = []
    fits_mask_list = []
    
    for r in remaining_rects:
        # Encontrar índice en lista original
        orig_idx = rects_originales.index(r)
        remaining_indices.append(orig_idx)
        
        # Calcular factibilidad SOLO para este espacio específico
        rw, rh = r
        _, _, sw, sh = active_space
        fits_normal = (rw <= sw and rh <= sh)
        fits_rotated = (rh <= sw and rw <= sh)
        feasible = fits_normal or fits_rotated
        fits_mask_list.append(1 if feasible else 0)

    if not any(fits_mask_list):
        return -1, None

    # Extraer embeddings pre-calculados solo para rectángulos remaining
    rect_enc = cached_rect_enc[:, remaining_indices, :]  # (1, N_remaining, d)
    rect_mask = torch.tensor([fits_mask_list], dtype=torch.bool, device=device)  # (1, N_remaining)
    
    # SOLO DECODER: Space features + decode step (con global_ctx dinámico)
    current_max_height = active_space[1] + active_space[3]  # y + h del espacio activo
    spaces_context = [active_space]  # Lista minimal para compute features
    space_feat_vec = st._space_features_optimized(
        active_space, container_width, Href, spaces_context, current_max_height
    )
    space_feat = torch.tensor(space_feat_vec, dtype=torch.float32, device=device).unsqueeze(0)
    
    step_idx = torch.tensor([step], dtype=torch.long, device=device)
    # decode_step ahora recalcula global_ctx dinámicamente basado en rect_mask
    probs, scores = model.decode_step(rect_enc, rect_mask, space_feat, step_idx)
    
    # Elegir el mejor factible
    probs_cpu = probs.squeeze(0).cpu()
    feasible_indices = [i for i, m in enumerate(fits_mask_list) if m == 1]
    if not feasible_indices:
        return -1, None
        
    feasible_probs = probs_cpu[feasible_indices]
    chosen_local = int(feasible_indices[int(torch.argmax(feasible_probs))])
    
    return chosen_local, remaining_rects[chosen_local]





@torch.no_grad()
def recursive_packing_pointer(
    space, 
    spaces, 
    rects, 
    placed, 
    estados=None, 
    Y_rect=None, 
    category="",
    model=None, 
    container_width=0, 
    device="cpu", 
    step_counter=None,
    rects_originales=None,
    cached_rect_enc=None,
    cached_global_ctx=None):
    """
    Versión pointer OPTIMIZADA de recursive_packing. 
    Usa embeddings pre-calculados del encoder para máxima eficiencia.
    """
    if step_counter is None:
        step_counter = [0]
    
    while rects:
        # Usar SOLO la versión OPTIMIZADA (decoder únicamente)
        rect_idx, rect = usar_modelo_pointer_para_decision_optimized(
            rects, space, model, container_width, category, device, step_counter[0],
            rects_originales, cached_rect_enc, cached_global_ctx
        )
        
        if rect_idx == -1 or rect is None:
            break
            
        # Verificar si el rectángulo cabe en el espacio
        fits, rotation = rect_fits_in_space(rect, space)
        if fits:
            # Rotar el rectángulo si es necesario
            if rotation == 1:
                rect = (rect[1], rect[0])

            ok, pos = place_rect(space, rect)
            if ok:
                placed.append((rect, pos))
                Y_rect.append(codificar_y_rect_con_lista_pointer(rects, rect_idx))

                # Dividir en nuevos S3 y S4 desde el subespacio actual
                S3, S4 = divide_space_2(space, rect, pos)
                temp_spaces = spaces.copy()
                temp_spaces.append(S3)
                temp_spaces.append(S4)

                # Eliminar rectángulo usado
                rects.pop(rect_idx)
                step_counter[0] += 1

                # Mostrar cuál espacio es más grande: S3 o S4
                area_S3 = S3[2] * S3[3]
                area_S4 = S4[2] * S4[3]
                
                if not rects:
                    return  

                if area_S3 > area_S4:
                    estado = st.codificar_estado(temp_spaces, rects, S3, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
                    estados.append(estado)

                    recursive_packing_pointer(S3, spaces, rects, placed, estados, Y_rect, category, model, container_width, device, step_counter, rects_originales, cached_rect_enc, cached_global_ctx)
                    temp_spaces.remove(S3)

                    if not rects:
                        return

                    recursive_packing_pointer(S4, spaces, rects, placed, estados, Y_rect, category, model, container_width, device, step_counter, rects_originales, cached_rect_enc, cached_global_ctx)
                    temp_spaces.remove(S4)
                else:
                    estado = st.codificar_estado(temp_spaces, rects, S4, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
                    estados.append(estado)

                    recursive_packing_pointer(S4, spaces, rects, placed, estados, Y_rect, category, model, container_width, device, step_counter, rects_originales, cached_rect_enc, cached_global_ctx)
                    temp_spaces.remove(S4)
                    
                    if not rects:
                        return

                    recursive_packing_pointer(S3, spaces, rects, placed, estados, Y_rect, category, model, container_width, device, step_counter, rects_originales, cached_rect_enc, cached_global_ctx)
                    temp_spaces.remove(S3)

                return
        break


@torch.no_grad()
def hr_packing_pointer(spaces, rects, category="", model=None, container_width=0, device="cpu"):
    """
    Versión pointer de hr_packing. Sigue EXACTAMENTE la misma estructura
    que hr_packing de hr_algorithm.py pero usa el modelo pointer para las decisiones.
    """
    placed = []
    rects1 = rects.copy()
    estados = []
    Y_rect = []
    step_counter = [0]
    
    # OPTIMIZACIÓN: Calcular encoder UNA SOLA VEZ al inicio
    rects_originales = rects.copy()  # Mantener referencia original para embeddings
    cached_rect_enc, cached_global_ctx = inicializar_cache_encoder(
        rects_originales, spaces, container_width, category, model, device
    )

    # Estado inicial
    estado = st.codificar_estado(spaces, rects1, spaces[0], cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
    estados.append(estado)

    while rects1:
        placed_flag = False

        for space in spaces:
            # Usar modelo pointer OPTIMIZADO (solo decoder)
            rect_idx, rect = usar_modelo_pointer_para_decision_optimized(
                rects1, space, model, container_width, category, device, step_counter[0],
                rects_originales, cached_rect_enc, cached_global_ctx
            )
            
            if rect_idx != -1 and rect is not None:
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    if ok:
                        temp_spaces = []
                        placed.append((rect, pos))

                        Y_rect.append(codificar_y_rect_con_lista_pointer(rects1, rect_idx))

                        # Dividir el espacio en S1 (encima, unbounded) y S2 (derecha, bounded)
                        S1, S2 = divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)

                        # Eliminar rectángulo insertado y espacio usado
                        rects1.pop(rect_idx)
                        spaces.remove(space)
                        step_counter[0] += 1

                        if rects1:  # Solo codificar el estado si quedan rectángulos
                            estado = st.codificar_estado(temp_spaces, rects1, S2, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
                            estados.append(estado)
                            
                        # Agregar S1 al espacio disponible para seguir iterando
                        spaces.append(S1)
                        # Llamar recursivamente a recursive_packing_pointer con S2 (bounded) + cache
                        recursive_packing_pointer(S2, spaces, rects1, placed, estados, Y_rect, category, model, container_width, device, step_counter, rects_originales, cached_rect_enc, cached_global_ctx)

                        placed_flag = True
                        break
                        
        if not placed_flag:
            break
            
    return placed, estados, Y_rect


def ordenar_por_area(rects):
    """Mantiene la misma función que hr_algorithm"""
    return hr.ordenar_por_area(rects)

def calcular_altura(placements):
    """Mantiene la misma función que hr_algorithm"""
    return hr.calcular_altura(placements)


@torch.no_grad()
def heuristic_recursion_pointer(rects, container_width, category="", model=None, device="cpu"):
    """
    Versión pointer de heuristic_recursion. Sigue EXACTAMENTE la misma estructura
    que heuristic_recursion de hr_algorithm.py pero usa el modelo pointer.
    
    NOTA: Por simplicidad, solo hace UNA ejecución (no permutaciones) ya que el modelo
    pointer debería ser más inteligente que probar permutaciones manualmente.
    Si quieres permutaciones como el HR original, descomenta el bucle doble.
    """
    # rects = ordenar_por_area(rects)  # Opcional: comentar si el modelo ya maneja el orden
    best_height = float('inf')
    best_placements = []
    rect_sequence = []

    all_states = []
    all_Y_rect = []
    best_placement_states = []
    best_placement_Y_states = []

    # Versión simple: una sola ejecución (el modelo decide todo)
    temp_rects = rects.copy()
    
    placements, estados, Y_rect = hr_packing_pointer(
        spaces=[(0, 0, container_width, 1000)],
        rects=temp_rects,
        category=category,
        model=model,
        container_width=container_width,
        device=device
    )
    
    used_heights = [pos[1] + rect[1] for rect, pos in placements]
    altura = max(used_heights) if used_heights else 0

    if altura < best_height:
        rect_sequence = rects.copy()  # Secuencia original
        best_height = altura
        best_placements = placements
        all_states = [estados]
        all_Y_rect = [Y_rect]
        best_placement_states = [estados]
        best_placement_Y_states = [Y_rect]

    # Si quieres probar permutaciones como el HR original, descomenta esto:
    """
    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            
            placements, estados, Y_rect = hr_packing_pointer(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects,
                category=category,
                model=model,
                container_width=container_width,
                device=device
            )
            
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0

            if altura < best_height:
                rect_sequence = temp_rects.copy()
                best_height = altura
                best_placements = placements
                all_states.append(estados)
                all_Y_rect.append(Y_rect)
                best_placement_states = [estados]
                best_placement_Y_states = [Y_rect]
    """

    return best_placements, best_height, rect_sequence, all_states, all_Y_rect, best_placement_states, best_placement_Y_states


# Mantener compatibilidad con la interfaz anterior
def heuristic_pointer_wrapper(
    rects: List[Rect],
    container_width: int,
    model: SPPPointerModel,
    category: str = "C1",
    device: str = "cpu",
):
    """Wrapper para mantener compatibilidad con run_pointer.py"""
    return heuristic_recursion_pointer(
        rects=rects,
        container_width=container_width,
        category=category,
        model=model,
        device=device
    )


# Reutilizar la función de visualización del HR original
def visualizar_packing(placements, container_width, container_height=None, show=True):
    """Usa directamente la función de hr_algorithm.py"""
    return hr.visualizar_packing(placements, container_width, container_height, show)


if __name__ == "__main__":
    # Pequeña prueba manual usando la nueva estructura OPTIMIZADA
    demo_rects = [(3, 5), (4, 4), (2, 7), (5, 2)]
    from pointer_model import SPPPointerModel
    
    model = SPPPointerModel()
    print("Ejecutando prueba con encoder OPTIMIZADO (calculado una sola vez)...")
    
    placements, altura, rect_sequence, all_states, all_Y_rect, best_placement_states, best_placement_Y_states = heuristic_recursion_pointer(
        demo_rects, 
        container_width=20, 
        model=model, 
        category="C1", 
        device="cpu"
    )
    print("Resultado demo pointer OPTIMIZADO:")
    print(f"  - Altura: {altura}")
    print(f"  - Placements: {placements}")
    print(f"  - Y_rect: {all_Y_rect}")
    print("✅ Encoder se calculó UNA SOLA VEZ, decoder se ejecutó en cada decisión")
