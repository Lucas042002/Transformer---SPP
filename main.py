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
placements, altura, rect_sequence = hr.heuristic_recursion(rectangles_C, container_width)

#print(f"Rectángulos: {rect_sequence}")
print(f"Altura final: {altura}")

# Visualizar el packing
hr.visualizar_packing(placements, container_width, container_height)
