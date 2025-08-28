import hr_algorithm as hr
import torch
import numpy as np
import states as st
import categories as cat

# ----------------------------
# Funciones del algoritmo HR (simplificado)
# ----------------------------


def codificar_estado(spaces, rects, espacio_seleccionado, W, Href=20):
    """
    Codifica el estado actual para el modelo
    """
    return st.codificar_estado(spaces, rects, espacio_seleccionado, W, Href, include_xy=True)

def rect_fits_in_space(rect, space):
    rw, rh = rect
    x, y, w, h = space

    # Prueba sin rotar y rotado
    # Sin rotar
    if rw <= w and rh <= h:
        return True, 0
    # Rotado
    if rh <= w and rw <= h:
        return True, 1
    return False, -1

def place_rect(space, rect):
    if rect_fits_in_space(rect, space):
        return True, (space[0], space[1])  # bottom-left
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

def procesar_estado_para_modelo(estado_raw, largo_max=18):
    """
    Convierte un estado raw a formato listo para el modelo (aplanado, sin seq_id)
    """
    # Aplanar directamente sin agregar seq_id
    state_flat = []
    for space in estado_raw[0]:  # Espacios
        state_flat.append(space)
    for rect in estado_raw[1]:   # Rectángulos
        state_flat.append(rect)
    
    # Padding
    while len(state_flat) < largo_max:
        state_flat.append([0] * 12)  # Vector de padding con 12 elementos

    return state_flat, estado_raw



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

def recursive_packing_con_modelo(space, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, device="cpu", acciones_modelo=None, logits_modelo=None):
    if not rects:
        return

    # El estado ya viene procesado desde la función principal
    estado_flat = estados[-1]  # Ya está aplanado y listo para usar

    encoder_input = torch.tensor([estado_flat], dtype=torch.float32).to(device)
    decoder_input = torch.tensor([[9]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(encoder_input, decoder_input)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy().flatten()
        
        if len(probs_np) > len(rects):
            probs_np[len(rects):] = -float('inf')
        action_idx = int(np.argmax(probs_np))

    acciones_modelo.append(action_idx)  
    logits_modelo.append(logits.cpu().numpy().flatten()) 


    # print(f"[Recursivo] Estado actual: {estado}")
    # print(f"Probabilidades de acciones: {probs.cpu().numpy().round(3).tolist()}")
    # print(f"Rectángulos disponibles: {rects}")
    # print(f"Índice elegido: {action_idx}")

    # Busca el rectángulo correspondiente a la acción
    if action_idx >= len(rects):
        return
    rect = rects[action_idx]
    fits, rotation = rect_fits_in_space(rect, space)
    
    if fits:
        if rotation == 1:
            rect = (rect[1], rect[0])
        ok, pos = place_rect(space, rect)

        if ok:
            placed.append((rect, pos))
            Y_rect.append(action_idx)  # Simplificación temporal
            print(f"Indice elegido: {action_idx}")

            S3, S4 = divide_space_2(space, rect, pos)
            temp_spaces = spaces.copy()
            temp_spaces.append(S3)
            temp_spaces.append(S4)

            rects.pop(action_idx)


            # Mostrar cuál espacio es más grande: S3 o S4
            area_S3 = S3[2] * S3[3]
            area_S4 = S4[2] * S4[3]

            if not rects:
                return

            if area_S3 > area_S4:
                estado_raw = codificar_estado(temp_spaces, rects, S3, container_width)  # container_width=20 por defecto
                state_flat_nuevo, _ = procesar_estado_para_modelo(estado_raw, largo_max)
                estados.append(state_flat_nuevo)
                print(f"Estado codificado (S3): {state_flat_nuevo}")
                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S3)

                if not rects:
                    return # Termina esta rama recursiva

                Y_rect.append(hr.codificar_y_rect_con_lista(rects, rect))
                print(f"Indice elegido: {hr.codificar_y_rect_con_lista(rects, rect)}")
                estado_raw = codificar_estado(temp_spaces, rects, S4, container_width)
                state_flat_nuevo, _ = procesar_estado_para_modelo(estado_raw, largo_max)
                estados.append(state_flat_nuevo)
                print(f"Estado codificado (S4): {state_flat_nuevo}")
                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S4)
            else:
                estado_raw = codificar_estado(temp_spaces, rects, S4, container_width)
                state_flat_nuevo, _ = procesar_estado_para_modelo(estado_raw, largo_max)
                estados.append(state_flat_nuevo)
                print(f"Estado codificado (S4): {state_flat_nuevo}")
                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S4)

                if not rects:
                    return  # Termina esta rama recursiva

                Y_rect.append(hr.codificar_y_rect_con_lista(rects, rect))
                print(f"Indice elegido: {hr.codificar_y_rect_con_lista(rects, rect)}")

                estado_raw = codificar_estado(temp_spaces, rects, S3, container_width)
                state_flat_nuevo, _ = procesar_estado_para_modelo(estado_raw, largo_max)
                estados.append(state_flat_nuevo)
                print(f"Estado codificado (S3): {state_flat_nuevo}")
                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S3)
            return  # Termina esta rama recursiva

def hr_packing_con_modelo(spaces, rects, model, device="cpu", container_width=0, category="C1"):
    placed = []
    rects1 = rects.copy()
    estados = []
    Y_rect = []
    acciones_modelo = []
    logits_modelo = []  
    largo_max = cat.CATEGORIES[category]["num_items"] + 1  # Tu modelo espera 18 elementos

    while rects1:
        colocado = False
        
        # Codificar estado actual
        estado_raw = codificar_estado(spaces, rects1, spaces[0], container_width)
        # print(f"Estado codificado raw: {len(estado_raw[0])} espacios, {len(estado_raw[1])} rects")
        
        # Procesar estado sin seq_id
        state_flat, estado_procesado = procesar_estado_para_modelo(estado_raw, largo_max)
        
        # print(f"State_flat length: {len(state_flat)}")
        # if len(state_flat) > 0:
        #     print(f"Primer elemento length: {len(state_flat[0])}")
        
        estados.append(state_flat)
        print(f"Estado codificado actual: {state_flat}")
        # Prepara el estado para el modelo encoder-decoder
        encoder_input = torch.tensor([state_flat], dtype=torch.float32).to(device)  # [1, 18, 12]
        decoder_input = torch.tensor([[9]], dtype=torch.long).to(device)  # [1, 1] start token
        
        with torch.no_grad():
            # Forward pass con ambos inputs
            logits = model(encoder_input, decoder_input)[:, -1, :]  # [1, num_classes]
            
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy().flatten()
            
            # Enmascara las clases no válidas
            if len(probs_np) > len(rects1):
                probs_np[len(rects1):] = -float('inf')
            action_idx = int(np.argmax(probs_np))
        
        acciones_modelo.append(action_idx)  
        logits_modelo.append(logits.cpu().numpy().flatten()) 

        # Resto de la lógica de colocación...
        if action_idx < len(rects1):
            rect = rects1[action_idx]
        # print(f"Rectángulo elegido: {rect}")
        # Busca un espacio donde quepa
            for space in spaces:
                fits, rotation = hr.rect_fits_in_space(rect, space)
                if fits:
                    if rotation == 1:
                        rect = (rect[1], rect[0])
                    ok, pos = hr.place_rect(space, rect)

                    if ok:
                        temp_spaces = []
                        placed.append((rect, pos))
                        
                        # Para Y_rect, usar el estado procesado
                        Y_rect.append(hr.codificar_y_rect_con_lista(rects, rect))
                        print(f"Indice elegido: {action_idx}")

                        S1, S2 = hr.divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)

                        rects1.pop(action_idx)
                        spaces.remove(space)

                        if rects1:
                            # Nuevo estado sin seq_id
                            estado_nuevo_raw = codificar_estado(temp_spaces, rects1, S2, container_width)
                            state_flat_nuevo, estado_nuevo_procesado = procesar_estado_para_modelo(
                                estado_nuevo_raw, largo_max
                            )
                            estados.append(state_flat_nuevo)
                        
                        spaces.append(S1)
                        recursive_packing_con_modelo(S2, spaces, rects1, placed, estados, Y_rect, largo_max, model, container_width, device, acciones_modelo, logits_modelo)

                        if rects1:
                            Y_rect.append(hr.codificar_y_rect_con_lista(rects1, rect))
                            print(f"Indice elegido: {hr.codificar_y_rect_con_lista(rects1, rect)}")

                            
                        colocado = True

                        # Estado final sin seq_id
                        estado_final_raw = codificar_estado(spaces, rects1, S1, container_width)
                        state_flat_final, _ = procesar_estado_para_modelo(
                            estado_final_raw, largo_max
                        )
                        estados.append(state_flat_final)
                        print(f"Estado codificado final: {state_flat_final}")
                        break
                        
        if not colocado:
            break
    return placed, estados, Y_rect, acciones_modelo, logits_modelo








def ordenar_por_area(rects):
    return sorted(rects, key=lambda r: r[0] * r[1], reverse=True)

def calcular_altura(placements):
    return max([y + h for (_, (x, y)), (w, h) in zip(placements, [p[0] for p in placements])], default=0)

def heuristic_recursion_transformer(rects, container_width, model, device="cpu", category="C1"):
    # rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    rect_sequence = []

    all_states = []
    all_Y_rect = []

    # for i in range(len(rects) - 1):
    #     for j in range(i + 1, len(rects)):
    temp_rects = rects.copy()
    #         print(f"Probando permutación: {i}, {j}")
    #         temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            
    try:
        placements, estados, Y_rect, acciones_modelo, logits_modelo = hr_packing_con_modelo(
            spaces=[(0, 0, container_width, 1000)],
            rects=temp_rects,
            model=model,
            device=device,
            container_width=container_width,
            category=category
        )
                
        used_heights = [pos[1] + rect[1] for rect, pos in placements]
        altura = max(used_heights) if used_heights else 0

        if altura <= best_height:
            rect_sequence = temp_rects.copy()
            best_height = altura
            best_placements = placements
        all_states.append(estados)
        all_Y_rect.append(Y_rect)
                
    except Exception as e:
        print(f"Error: {e}")
        

    return best_placements, best_height, rect_sequence, all_states, all_Y_rect





