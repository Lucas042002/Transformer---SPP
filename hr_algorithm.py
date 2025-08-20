import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import test as tst

CATEGORIES = {
    "C1": {"num_items": 17, "width": 20, "height": 20},
    "C2": {"num_items": 25, "width": 40, "height": 15},
    "C3": {"num_items": 29, "width": 60, "height": 30},
    "C4": {"num_items": 49, "width": 60, "height": 60},
    "C5": {"num_items": 73, "width": 60, "height": 90},
    "C6": {"num_items": 97, "width": 80, "height": 120},
    "C7": {"num_items": 197, "width": 160, "height": 240},
}

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
def codificar_estado(spaces, rects, espacio_seleccionado):
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
    return estado


def codificar_y_estado(estado, rect):
    """
    Devuelve el índice entero (int) dentro de `estado` donde se encuentra
    el rectángulo `rect` seleccionado por el HR en este paso.
    Si no se encuentra una coincidencia válida, devuelve -1.
    """
    # Buscar el primer bloque pendiente (contexto==0) que coincida con rect
    print(f"Buscando rectángulo {rect} en estado {estado} elementos\n")

    for idx, v in enumerate(estado):
        # v tiene la forma [h, w, area, a_utilizar, contexto]
        # comparar con rect=(w,h) o (h,w) por seguridad
        if ((v[1], v[0]) == rect or (v[0], v[1]) == rect):
            return int(idx)
    # Si no se encontró coincidencia, devolver -1
    return -1

def codificar_y_rect_con_lista(rects_pendientes, rect):
    """
    rects_pendientes: lista [(w,h), ...] en el orden en que armaste R_in[0]
    rect: (w,h) elegido por HR
    Devuelve one-hot de largo N.
    """
    N = len(rects_pendientes)
    Y = [0] * N

    for idx, r in enumerate(rects_pendientes):
        if r == rect or r == (rect[1], rect[0]):
            Y[idx] = 1
            break

    return Y


def recursive_packing(space, spaces, rects, placed, estados=None, Y_rect=None, category=""):
    while rects:
        for i, rect in enumerate(rects):
            # Verificar si el rectángulo cabe en el espacio
            fits, rotation = rect_fits_in_space(rect, space)
            if fits:
                # Rotar el rectángulo si es necesario
                if rotation == 1:
                    rect = (rect[1], rect[0])
        

                ok, pos = place_rect(space, rect)
                # print(f"[Recursivo] Probando rectángulo {rect} en espacio {space} -> posición {pos}")
                if ok:
                    placed.append((rect, pos))
                    Y_rect.append(codificar_y_rect_con_lista(rects, rect))

                    # print(f"[Recursivo] Rectángulo {rect} colocado en ({space[2]},{space[3]})")
                    # Dividir en nuevos S3 y S4 desde el subespacio actual
                    S3, S4 = divide_space_2(space, rect, pos)
                    # print(f"[Recursivo] División: S3={S3}, S4={S4}\n")
                    temp_spaces = spaces.copy()
                    temp_spaces.append(S3)
                    temp_spaces.append(S4)

                    # Y_rect.append(rect)

                    # Eliminar rectángulo usado
                    rects.pop(i)
                    # Mostrar cuál espacio es más grande: S3 o S4
                    area_S3 = S3[2] * S3[3]
                    area_S4 = S4[2] * S4[3]
                    
                    if not rects:
                        return  

                    # print(f"Área S3: {area_S3}, {S3[2]},{S3[3]} Área S4: {area_S4}, {S4[2]},{S4[3]}")
                    if area_S3 > area_S4:
                        estado = tst.codificar_estado(temp_spaces, rects, S3, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                        estados.append(estado)

                        recursive_packing(S3, spaces, rects, placed, estados, Y_rect, category)
                        temp_spaces.remove(S3)

                        if not rects:
                            return # Termina esta rama recursiva

                        Y_rect.append(codificar_y_rect_con_lista(rects, rect))
                        estado = tst.codificar_estado(temp_spaces, rects, S4, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                        estados.append(estado)

                        recursive_packing(S4, spaces, rects, placed, estados, Y_rect, category)
                        temp_spaces.remove(S4)
                    else:
                        estado = tst.codificar_estado(temp_spaces, rects, S4, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                        estados.append(estado)

                        recursive_packing(S4, spaces, rects, placed, estados, Y_rect, category)
                        temp_spaces.remove(S4)
                        
                        if not rects:
                            return  # Termina esta rama recursiva

                        Y_rect.append(codificar_y_rect_con_lista(rects, rect))
                        estado = tst.codificar_estado(temp_spaces, rects, S3, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                        estados.append(estado)

                        recursive_packing(S3, spaces, rects, placed, estados, Y_rect, category)
                        temp_spaces.remove(S3)

                    return  # Termina esta rama recursiva
        break  # Si ningún rectángulo cabe, se termina

def hr_packing(spaces, rects, category=""):
    placed = []
    rects1 = rects.copy()

    estados = []  # Lista para almacenar los estados codificados
    Y_rect = []  # Lista para almacenar los índices de los rectángulos elegidos
    estado = tst.codificar_estado(spaces, rects1, spaces[0], CATEGORIES[category]['width'], CATEGORIES[category]['height'])  # Estado inicial
    estados.append(estado)  # Estado inicial

    while rects1:
        placed_flag = False

        for space in spaces:
            for i, rect in enumerate(rects1):
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    # print(f"Probando rectángulo {rect} en espacio {space} -> posición {pos}")
                    if ok:

                        #temp_spaces = spaces.copy() 
                        temp_spaces = []
                        placed.append((rect, pos))
                        # print(f"Rectángulo {rect} colocado en ({space[2]},{space[3]})")

                        Y_rect.append(codificar_y_rect_con_lista(rects1, rect))


                        # Dividir el espacio en S1 (encima, unbounded) y S2 (derecha, bounded)
                        S1, S2 = divide_space(space, rect, pos)
                        temp_spaces.append(S1)
                        temp_spaces.append(S2)
                        # print(f"Dividiendo espacio {space} en S1={S1} (encima) y S2={S2} (derecha)\n")

                        # Eliminar rectángulo insertado y espacio usado
                        rects1.pop(i)
                        spaces.remove(space)

                        if rects1:  # Solo codificar el estado si quedan rectángulos
                            estado = tst.codificar_estado(temp_spaces, rects1, S2, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                            estados.append(estado)
                            
                        # Agregar S1 al espacio disponible para seguir iterando
                        spaces.append(S1)
                        # Llamar recursivamente a RecursivePacking con S2 (bounded)
                        # temp_spaces.remove(S2)

                        recursive_packing(S2, spaces, rects1, placed, estados, Y_rect, category)
                        if rects1:
                            Y_rect.append(codificar_y_rect_con_lista(rects1, rect))

                        placed_flag = True

                        estado = tst.codificar_estado(spaces, rects1, S1, CATEGORIES[category]['width'], CATEGORIES[category]['height'])
                        estados.append(estado)
                        
                        break  
            if placed_flag:
                break
        if not placed_flag:
            break
    Y_rect.append(codificar_y_rect_con_lista(rects1, rect))
    return placed, estados, Y_rect

# ----------------------------
def ordenar_por_area(rects):
    return sorted(rects, key=lambda r: r[0] * r[1], reverse=True)

def calcular_altura(placements):
    return max([y + h for (_, (x, y)), (w, h) in zip(placements, [p[0] for p in placements])], default=0)

def heuristic_recursion(rects, container_width, category = ""):
    rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    rect_sequence = []

    all_states = []
    all_Y_rect = []

    best_placement_states = []
    best_placement_Y_states = []

    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]
            # print(f"Probando permutación: {temp_rects}")
            placements, estados, Y_rect = hr_packing(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects,
                category=category
            )
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0

            if altura < best_height:
                # print(f"Mejor altura encontrada: {altura} con rectángulos {temp_rects}")
                rect_sequence = temp_rects.copy()
                best_height = altura
                best_placements = placements
                all_states.append(estados)
                all_Y_rect.append(Y_rect)
                best_placement_states = estados
                best_placement_Y_states = Y_rect

    

    return best_placements, best_height, rect_sequence, all_states, all_Y_rect, best_placement_states, best_placement_Y_states
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
