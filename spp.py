import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualizar_packing(placements, container_width, container_height=None):
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
    plt.show()



# ----------------------------
# 1. Datos del problema
# ----------------------------
container_width = 160
container_height = 240  # referencia, no se impone como límite


# Leer rectángulos desde el archivo c7p2.txt
rectangles_c1p1 = []
with open("c7p2.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            height, width = map(int, parts[:2])
            rectangles_c1p1.append((width, height))  # (ancho, alto)

# Rotar todos los rectángulos para que sean más anchos que altos
# rectangles_c1p1 = [(max(w, h), min(w, h)) for (w, h) in rectangles_c1p1]

initial_space = [(0, 0, container_width, 1000)]  # altura infinita simulada

# ----------------------------
# 2. Funciones del algoritmo HR (simplificado)
# ----------------------------
def rect_fits_in_space(rect, space):
    rw, rh = rect
    x, y, w, h = space
    return rw <= w and rh <= h

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

def recursive_packing(space, spaces, rects, placed):
    while rects:
        for i, rect in enumerate(rects):
            if rect_fits_in_space(rect, space):
                ok, pos = place_rect(space, rect)
                # print(f"[Recursivo] Probando rectángulo {rect} en espacio {space} -> posición {pos}")
                if ok:
                    placed.append((rect, pos))
                    # print(f"[Recursivo] Rectángulo {rect} colocado en {pos}")

                    # Dividir en nuevos S3 y S4 desde el subespacio actual
                    S3, S4 = divide_space(space, rect, pos)
                    # print(f"[Recursivo] División: S3={S3}, S4={S4}\n")

                    # Eliminar rectángulo usado
                    rects.pop(i)
                    # Mostrar cuál espacio es más grande: S3 o S4
                    area_S3 = S3[2] * S3[3]
                    area_S4 = S4[2] * S4[3]

                    # print(f"Área S3: {area_S3}, {S3[2]},{S3[3]} Área S4: {area_S4}, {S4[2]},{S4[3]}")
                    if area_S3 > area_S4:
                        recursive_packing(S3, spaces, rects, placed)
                        recursive_packing(S4, spaces, rects, placed)
                    else:
                        recursive_packing(S4, spaces, rects, placed)
                        recursive_packing(S3, spaces, rects, placed)
                

                    return  # Termina esta rama recursiva
        break  # Si ningún rectángulo cabe, se termina

def hr_packing(spaces, rects):
    placed = []
    rects1 = rects.copy()

    while rects1:
        placed_flag = False

        for space in spaces:
            for i, rect in enumerate(rects1):
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    # print(f"Probando rectángulo {rect} en espacio {space} -> posición {pos}")
                    if ok:
                        placed.append((rect, pos))
                        # print(f"Rectángulo {rect} colocado en {pos}")

                        # Dividir el espacio en S1 (encima, unbounded) y S2 (derecha, bounded)
                        S1, S2 = divide_space(space, rect, pos)
                        # print(f"Dividiendo espacio {space} en S1={S1} (encima) y S2={S2} (derecha)\n")

                        # Eliminar rectángulo insertado y espacio usado
                        rects1.pop(i)
                        spaces.remove(space)

                        # Agregar S1 al espacio disponible para seguir iterando
                        spaces.append(S1)

                        # Llamar recursivamente a RecursivePacking con S2 (bounded)
                        recursive_packing(S2, spaces, rects1, placed)

                        placed_flag = True
                        break
            if placed_flag:
                break

        if not placed_flag:
            break

    return placed


# ----------------------------

def ordenar_por_area(rects):
    return sorted(rects, key=lambda r: r[0] * r[1], reverse=True)

def calcular_altura(placements):
    return max([y + h for (_, (x, y)), (w, h) in zip(placements, [p[0] for p in placements])], default=0)

def heuristic_recursion(rects, container_width):
    rects = ordenar_por_area(rects)
    best_height = float('inf')
    best_placements = []
    # print(f"Rectángulos ordenados por área: {rects}")
    # intercambiar pares (i, j) para generar permutaciones locales
    for i in range(len(rects) - 1):
        for j in range(i + 1, len(rects)):
            temp_rects = rects.copy()
            temp_rects[i], temp_rects[j] = temp_rects[j], temp_rects[i]

            # print(f"Probando permutación: {temp_rects}")    
            placements = hr_packing(
                spaces=[(0, 0, container_width, 1000)],
                rects=temp_rects
            )
            used_heights = [pos[1] + rect[1] for rect, pos in placements]
            altura = max(used_heights) if used_heights else 0
            
            if altura <= best_height:
                best_height = altura
                best_placements = placements

    return best_placements, best_height

# ----------------------------
# 3. Ejecutar
# ----------------------------
placements, altura = heuristic_recursion(rectangles_c1p1, container_width)

print(f"Altura final: {altura}")
visualizar_packing(placements, container_width, container_height)
