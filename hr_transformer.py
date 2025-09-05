import hr_algorithm as hr
import torch
import numpy as np
import states as st
import categories as cat

# ----------------------------
# Funciones del algoritmo HR (simplificado)
# ----------------------------

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

def procesar_estado_para_modelo(estado_raw, largo_max=18): #Funciona
    """
    Convierte un estado raw a formato listo para el modelo (aplanado, sin seq_id)
    """
    # Aplanar directamente sin agregar seq_id
    state_flat = []
    for space in estado_raw[0]:  # Espacios
        state_flat.append(space)
    for rect in estado_raw[1]:   # Rectángulos
        state_flat.append(rect)
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

def recursive_packing_con_modelo(space, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, category, device="cpu", acciones_modelo=None, logits_modelo=None):
    if not rects:
        return

    # El estado ya viene procesado desde la función principal
    estado_flat = estados[-1]  # Ya está aplanado y listo para usar
    print(f"Estado actual (flat): {estado_flat}")
    encoder_input = torch.tensor([estado_flat], dtype=torch.float32).to(device)
    # Antes de llamar al modelo:
    decoder_seq = [START_TOKEN] + acciones_modelo  # historial de acciones previas
    decoder_input = torch.tensor([decoder_seq], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(encoder_input, decoder_input)[:, -1, :]
        
        # CORRECCIÓN: Enmascarar correctamente las acciones inválidas
        if len(rects) < logits.size(-1):
            # Crear máscara para acciones válidas
            mask = torch.full_like(logits, -float('inf'))
            mask[0, :len(rects)] = logits[0, :len(rects)]
            logits = mask
        
        probs = torch.softmax(logits, dim=-1)
        print(f"{probs}")
        action_idx = int(torch.argmax(probs, dim=-1))
        
        # Validación adicional
        if action_idx >= len(rects):
            print(f"WARNING: Modelo predijo índice {action_idx} pero solo hay {len(rects)} rectángulos")
            action_idx = 0  # Fallback

        # print(f"DEBUG: rects={len(rects)}, logits_shape={logits.shape}, action_idx={action_idx}")



    acciones_modelo.append(action_idx)  
    logits_modelo.append(logits.cpu().numpy().flatten()) 


    # print(f"[Recursivo] Estado actual: {estado}")
    # print(f"Probabilidades de acciones: {probs.cpu().numpy().round(3).tolist()}")
    # print(f"Rectángulos disponibles: {rects}")
    # print(f"Índice elegido: {action_idx}")

    # Busca el rectángulo correspondiente a la acción
    if action_idx > len(rects):
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
            
            #print(f"Indice elegido: {action_idx}")

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
                estado = st.codificar_estado(temp_spaces, rects, S3, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])  # container_width=20 por defecto
                estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
                estados.append(estado_flat)
                # print(f"Estado codificado (S3): {estado}")
                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, category, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S3)

                if not rects:
                    return # Termina esta rama recursiva

                # Y_rect.append(hr.codificar_y_rect_con_lista(rects, rect))

                # print(f"Indice elegido: {hr.codificar_y_rect_con_lista(rects, rect)}")
                estado = st.codificar_estado(temp_spaces, rects, S4, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])  # container_width=20 por defecto
                estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
                # print(f"Estado codificado (S4): {estado}")
                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, category, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S4)
            else:
                estado = st.codificar_estado(temp_spaces, rects, S4, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
                estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
                # print(f"Estado codificado (S4): {estado}")
                estados.append(estado_flat)
                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, category, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S4)

                if not rects:
                    return  # Termina esta rama recursiva

                # Y_rect.append(hr.codificar_y_rect_con_lista(rects, rect))

                estado = st.codificar_estado(temp_spaces, rects, S3, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])  # container_width=20 por defecto
                estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
                # print(f"Estado codificado (S3): {estado}")
                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model, container_width, category, device, acciones_modelo, logits_modelo)
                temp_spaces.remove(S3)
            return  # Termina esta rama recursiva



# En hr_transformer.py, actualizar la función hr_packing_con_modelo para retornar logits:


START_TOKEN = 19
END_TOKEN = 18

def hr_packing_con_modelo(spaces, rects, model, device="cpu", container_width=0, category="C1"):
    placed = []
    rects1 = rects.copy()
    estados = []
    Y_rect = []
    acciones_modelo = []
    logits_modelo = []  
    largo_max = cat.CATEGORIES[category]["num_items"] + 1


    while rects1:
        colocado = False
        
        # Codificar estado actual
        estado = st.codificar_estado(spaces, rects1, spaces[-1], cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
        estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
        estados.append(estado_flat)

        print(f"Estado actual (flat)1: {estado_flat}")

        # Prepara el estado para el modelo encoder-decoder
        encoder_input = torch.tensor([estado_flat], dtype=torch.float32).to(device)
        decoder_seq = [START_TOKEN] + acciones_modelo  # historial de acciones previas
        decoder_input = torch.tensor([decoder_seq], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(encoder_input, decoder_input)[:, -1, :]
            
            # CORRECCIÓN: Enmascarar correctamente las acciones inválidas
            if len(rects1) < logits.size(-1):
                # Crear máscara para acciones válidas
                mask = torch.full_like(logits, -float('inf'))
                mask[0, :len(rects1)] = logits[0, :len(rects1)]
                logits = mask
            
            probs = torch.softmax(logits, dim=-1)
            action_idx = int(torch.argmax(probs, dim=-1))
            
            # Validación adicional
            if action_idx >= len(rects1):
                print(f"WARNING: Modelo predijo índice {action_idx} pero solo hay {len(rects1)} rectángulos")
                action_idx = 0  # Fallback
            
            # print(f"DEBUG: rects1={len(rects1)}, logits_shape={logits.shape}, action_idx={action_idx}")
        
        
        # Guardar logits para métricas
        logits_modelo.append(logits.cpu().numpy().flatten())
        
        # Si el modelo predice end_token, termina el ciclo
        if action_idx >= len(rects1):
            break

        acciones_modelo.append(action_idx)
        
        if action_idx < len(rects1):
            rect = rects1[action_idx]
            
            for space in spaces:
                fits, rotation = hr.rect_fits_in_space(rect, space)
                if fits:
                    if rotation == 1:
                        rect = (rect[1], rect[0])
                    ok, pos = hr.place_rect(space, rect)

                    if ok:
                        temp_spaces = []
                        placed.append((rect, pos))
                        Y_rect.append(action_idx)  # Usar índice directo

                        S1, S2 = hr.divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)

                        rects1.pop(action_idx)
                        spaces.remove(space)

                        if rects1:
                            estado = st.codificar_estado(temp_spaces, rects1, S2, cat.CATEGORIES[category]['width'], cat.CATEGORIES[category]['height'])
                            estado_flat, _ = procesar_estado_para_modelo(estado, largo_max)
                            estados.append(estado_flat)
                        else:
                            break
                        
                        spaces.append(S1)
                        recursive_packing_con_modelo(S2, spaces, rects1, placed, estados, Y_rect, largo_max, model, container_width, category, device, acciones_modelo, logits_modelo)

                        colocado = True
                        break
                        
        if not colocado:
            break
            
    return placed, estados, Y_rect, acciones_modelo, logits_modelo



def heuristic_recursion_transformer(rects, container_width, model, device="cpu", category="C1"):
    # rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    rect_sequence = []

    all_states = []
    all_Y_rect = []

    temp_rects = rects.copy()

    # try:
    placements, estados, Y_rect, acciones_modelo, logits_modelo = hr_packing_con_modelo(
        spaces=[(0, 0, container_width, 1000)],
        rects=temp_rects,
        model=model,
        device=device,
        container_width=container_width,
        category=category
    )
    # print(f"Colocaciones realizadas: {placements}")

    used_heights = [pos[1] + rect[1] for rect, pos in placements]
    altura = max(used_heights) if used_heights else 0

    if altura <= best_height:
        rect_sequence = temp_rects.copy()
        best_height = altura
        best_placements = placements
    all_states.append(estados)
    all_Y_rect.extend(Y_rect)
                
    # except Exception as e:
    #     print(f"Error: {e}")
        

    return best_placements, best_height, rect_sequence, all_states, all_Y_rect





