import hr_algorithm as hr
import torch

# ----------------------------
# Funciones del algoritmo HR (simplificado)
# ----------------------------

def codificar_estado(spaces, rects, espacio_seleccionado, largo_max):
    """
    Codifica el estado actual como una lista de vectores [h, w, area, a_utilizar, contexto].
    - spaces: lista de subespacios disponibles [(x, y, w, h), ...]
    - rects: lista de bloques pendientes [(w, h), ...]
    - espacio_seleccionado: tupla (x, y, w, h) del espacio seleccionado para colocar un rectángulo.
    """
    # Buscar el índice del espacio seleccionado (match exacto)
    idx_seleccionado = -1
    for idx, s in enumerate(spaces):
        if s == espacio_seleccionado:
            idx_seleccionado = idx
            break

    estado = []
    # Codificar subespacios
    for idx, (x, y, w, h) in enumerate(spaces):
        area = w * h
        a_utilizar = 1 if idx == idx_seleccionado else 0
        contexto = 1  # Es un subespacio utilizable
        estado.append([h, w, area, a_utilizar, contexto])
    # Codificar bloques pendientes
    for (w, h) in rects:
        area = w * h
        a_utilizar = 0
        contexto = 0  # Es un bloque pendiente, no un subespacio
        estado.append([h, w, area, a_utilizar, contexto])

    # Asegurarse de que el estado tenga largo máximo
    while len(estado) < largo_max:
        estado.append([0, 0, 0, 0, 0])  # Rellenar con ceros

    return estado




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

def recursive_packing_con_modelo(space, spaces, rects, placed, estados, Y_rect, largo_max ,model, device="cpu"):
    if not rects:
        return

    # Codifica el estado actual
    estado = estados[-1] 

    # Prepara el estado para el modelo
    x = torch.tensor([estado], dtype=torch.float32).to(device)  # [1, seq_len, features]
    with torch.no_grad():
        logits = model(x)  # [1, seq_len, num_classes]
        probs = torch.softmax(logits, dim=-1)
        action = probs[0, -1].argmax().item()  # O ajusta según tu output
    
    print(f"Estado actual: {estado}")
    print(f"Probabilidades de acciones: {probs[0, -1].tolist()}")

    # Busca el rectángulo correspondiente a la acción
    if action >= len(rects):
        return  # Acción inválida
    rect = rects[action]
    print(f"Intentando colocar rectángulo {rect} (acción {action})")

    fits, rotation = rect_fits_in_space(rect, space)
    if fits:

        # Rotar el rectángulo si es necesario
        if rotation == 1:
            rect = (rect[1], rect[0])
        ok, pos = place_rect(space, rect)

        if ok:

            placed.append((rect, pos))
            Y_rect.append(hr.codificar_y_estado(estados[-1], rect))
            # print(f"[Recursivo] Rectángulo {rect} colocado en ({space[2]},{space[3]})")
            
            S3, S4 = divide_space_2(space, rect, pos)
            # print(f"[Recursivo] División: S3={S3}, S4={S4}\n")
            temp_spaces = spaces.copy()
            temp_spaces.append(S3)
            temp_spaces.append(S4)

            # Y_rect.append(rect)
            # Eliminar rectángulo usado
            rects.pop(action)


            # Mostrar cuál espacio es más grande: S3 o S4
            area_S3 = S3[2] * S3[3]
            area_S4 = S4[2] * S4[3]

            if not rects:
                return

            if area_S3 > area_S4:
                estado = codificar_estado(temp_spaces, rects, S3, largo_max)
                estados.append(estado)

                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model)
                temp_spaces.remove(S3)

                if not rects:
                    return # Termina esta rama recursiva

                Y_rect.append(hr.codificar_y_estado(estados[-1], rect))
                estado = codificar_estado(temp_spaces, rects, S4, largo_max)
                estados.append(estado)

                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model)
                temp_spaces.remove(S4)
            else:
                estado = codificar_estado(temp_spaces, rects, S4, largo_max)
                estados.append(estado)

                recursive_packing_con_modelo(S4, spaces, rects, placed, estados, Y_rect, largo_max, model)
                temp_spaces.remove(S4)

                if not rects:
                    return  # Termina esta rama recursiva

                Y_rect.append(hr.codificar_y_estado(estados[-1], rect))
                estado = codificar_estado(temp_spaces, rects, S3, largo_max)
                estados.append(estado)

                recursive_packing_con_modelo(S3, spaces, rects, placed, estados, Y_rect, largo_max, model)
                temp_spaces.remove(S3)
            return  # Termina esta rama recursiva




def hr_packing_con_modelo(spaces, rects, model, device="cpu"):
    placed = []
    rects1 = rects.copy()
    estados = []
    Y_rect = []
    largo_max = len(rects) + 1  # Largo máximo del estado
    estado = codificar_estado(spaces, rects1, spaces[0],largo_max )

    estados.append(estado)

    while rects1:
        # Prepara el estado para el modelo
        x = torch.tensor([estado], dtype=torch.float32).to(device)  # [1, seq_len, features]
        with torch.no_grad():
            logits = model(x)  # [1, seq_len, num_classes]
            probs = torch.softmax(logits, dim=-1)
            # Selecciona la acción con mayor probabilidad
            action = probs[0, -1].argmax().item()  # O ajusta según tu output

        print(f"Estado actual: {estado}")
        print(f"Probabilidades de acciones: {probs[0, -1].tolist()}")
        # Busca el rectángulo correspondiente a la acción
        # (Aquí debes mapear el índice de acción al rectángulo pendiente)
        # Por ejemplo, si tus acciones son índices de rectángulos pendientes:

        if action >= len(rects1):
            break  # Acción inválida
        rect = rects1[action]

        print(f"Intentando colocar rectángulo {rect} (acción {action})")

        # Busca un espacio donde quepa
        colocado = False

        for space in spaces:

            fits, rotation = hr.rect_fits_in_space(rect, space)
            print(f"Probando espacio {space} para rectángulo {rect}: {'Encaja' if fits else 'No encaja'} (rotación: {rotation})")
            if fits:

                if rotation == 1:
                    rect = (rect[1], rect[0])
                ok, pos = hr.place_rect(space, rect)

                if ok:

                    temp_spaces = []
                    placed.append((rect, pos))
                    # print(f"Rectángulo {rect} colocado en ({space[2]},{space[3]})")

                    Y_rect.append(hr.codificar_y_estado(estado, rect))

                    # Dividir el espacio en S1 (encima, unbounded) y S2 (derecha, bounded)

                    S1, S2 = hr.divide_space(space, rect, pos)
                    temp_spaces.append(S1)
                    temp_spaces.append(S2)

                    # print(f"Dividiendo espacio {space} en S1={S1} (encima) y S2={S2} (derecha)\n")
                    # Eliminar rectángulo insertado y espacio usado
                    rects1.pop(action)
                    spaces.remove(space)

                    if rects1:  # Solo codificar el estado si quedan rectángulos
                        estado = codificar_estado(temp_spaces, rects1, S2, largo_max)
                        estados.append(estado)
                        
                    # Agregar S1 al espacio disponible para seguir iterando
                    spaces.append(S1)
                    # Llamar recursivamente a RecursivePacking con S2 (bounded)
                    recursive_packing_con_modelo(S2, spaces, rects1, placed, estados, Y_rect, largo_max, model)

                    if rects1:
                        Y_rect.append(hr.codificar_y_estado(estados[-1], rect))
                        
                    colocado = True

                    estado = codificar_estado(spaces, rects1, S1, largo_max)
                    estados.append(estado)

                    break
        if not colocado:
            break
    return placed, estados, Y_rect


def ordenar_por_area(rects):
    return sorted(rects, key=lambda r: r[0] * r[1], reverse=True)

def calcular_altura(placements):
    return max([y + h for (_, (x, y)), (w, h) in zip(placements, [p[0] for p in placements])], default=0)


def heuristic_recursion_transformer(rects, container_width, model, device="cpu"):
    rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    rect_sequence = []

    all_states = []
    all_Y_rect = []

    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            # print(f"Probando permutación: {temp_rects}")
            placements, estados, Y_rect = hr_packing_con_modelo(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects,
                model=model,
                device=device
            )
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0

            if altura <= best_height:
                # print(f"Mejor altura encontrada: {altura} con rectángulos {temp_rects}")
                rect_sequence = temp_rects.copy()
                best_height = altura
                best_placements = placements


            all_states.append(estados)
            all_Y_rect.append(Y_rect)
    

    return best_placements, best_height, rect_sequence, all_states, all_Y_rect






