import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def visualizar_packing(placements, container_width, container_height=None):
    fig, ax = plt.subplots()
    colors = {}

    used_heights = [pos[1] + rect[1] for rect, pos in placements]
    max_height = max(used_heights) if used_heights else 0
    container_height = container_height or max_height

    # Dibujar contenedor
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, container_height)
    ax.set_aspect('equal')
    ax.set_title("Visualización del HR Packing")
    ax.set_xlabel("Ancho")
    ax.set_ylabel("Altura")

    for i, (rect, pos) in enumerate(placements):
        w, h = rect
        x, y = pos
        color = colors.get(rect)
        if color is None:
            color = [random.random() for _ in range(3)]
            colors[rect] = color
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color, label=f"{rect}"))
        ax.text(x + w/2, y + h/2, f"{i}", fontsize=8, ha='center', va='center', color='black')

    plt.grid(True)
    plt.tight_layout()
    plt.show()




# ----------------------------
# 1. Datos del problema
# ----------------------------
container_width = 20
container_height = 20  # referencia, no se impone como límite

rectangles_c1p1 = [
    (2, 12), (7, 12), (8, 6), (3, 6),
    (3, 5), (5, 5), (3, 12), (3, 7),
    (5, 7), (2, 6), (3, 2), (4, 2),
    (3, 4), (4, 4), (9, 2), (11, 2)
]

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
    x, y, w, h = space
    rw, rh = rect
    rx, ry = pos
    s1 = (rx + rw, y, x + w - (rx + rw), h)
    s2 = (x, ry + rh, rw, y + h - (ry + rh))
    return s1, s2

def hr_packing(spaces, rects):
    placed = []
    rects = rects.copy()
    while rects:
        placed_flag = False
        for space in spaces:
            for i, rect in enumerate(rects):
                if rect_fits_in_space(rect, space):
                    ok, pos = place_rect(space, rect)
                    if ok:
                        placed.append((rect, pos))
                        s1, s2 = divide_space(space, rect, pos)
                        rects.pop(i)
                        spaces.remove(space)
                        spaces.extend([s1, s2])
                        placed_flag = True
                        break
            if placed_flag:
                break
        if not placed_flag:
            break
    return placed

# ----------------------------
# 3. Ejecutar
# ----------------------------
placements = hr_packing(initial_space, rectangles_c1p1)

print("\nResultado de HR Packing:")
for rect, pos in placements:
    print(f"Rectángulo {rect} colocado en {pos}")

visualizar_packing(placements, container_width)
