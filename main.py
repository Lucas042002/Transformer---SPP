import hr_algorithm as hr

# ----------------------------
# Datos del problema
# ----------------------------

# Leer rectángulos desde el archivo correspondiente
rectangles_C = []
with open("c1p1.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            height, width = map(int, parts[:2])
            rectangles_C.append((width, height))  # (ancho, alto)

# Contenedores por categoria, Descomenta la categoría que deseas usar: (ancho, alto)
container_width, container_height = 20, 20    # c1
# container_width, container_height = 60, 30    # c3
# container_width, container_height = 60, 60    # c4
# container_width, container_height = 90, 60    # c5
# container_width, container_height = 120, 80   # c6
# container_width, container_height = 160, 240  # c7

initial_space = [(0, 0, container_width, 1000)]  # altura infinita simulada

# ----------------------------
# Ejecutar
# ----------------------------
result = hr.heuristic_recursion(rectangles_C, container_width)

# Compatibilidad con ambas versiones de heuristic_recursion
if len(result) == 5:
    placements, altura, rect_sequence, all_states, all_Y_rect = result
elif len(result) == 4:
    placements, altura, rect_sequence, all_states = result
    all_Y_rect = None
else:
    placements, altura, rect_sequence = result
    all_states = None
    all_Y_rect = None

#print(f"Rectángulos: {rect_sequence}")
print(f"Altura final: {altura}")

# Mostrar estados y los índices de los rectángulos elegidos si están disponibles
if all_states is not None:
    print(f"\nCantidad de secuencias de estados generadas: {len(all_states)}")
    idx = len(all_states) - 1
    print(f"\nSecuencia de estados para la última permutación ({idx+1}):")
    for paso, estado in enumerate(all_states[idx]):
        print(f"  Paso {paso+1}: {estado}")
    if all_Y_rect is not None:
        print(f"  Índices de rectángulos elegidos (Y_rect): {all_Y_rect[idx]}")

# Visualizar el packing
hr.visualizar_packing(placements, container_width, container_height)