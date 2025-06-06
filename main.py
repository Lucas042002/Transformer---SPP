import hr_algorithm as hr
import transformer as tr
from sklearn.model_selection import train_test_split

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
# Visualizar el packing
# hr.visualizar_packing(placements, container_width, container_height)

# Mostrar estados y los índices de los rectángulos elegidos si están disponibles
if all_states is not None:
    print(f"\nCantidad de secuencias de estados generadas: {len(all_states)}")

# Preparar los datos para el modelo
X, Y = tr.pad_and_prepare_data(all_states, all_Y_rect)

# División entrenamiento/validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Crear el modelo
state_dim = X.shape[2]  # Dimensión del vector de estado
num_actions = Y.shape[2]  # Número de acciones posibles
max_seq_len = X.shape[1]  # Longitud máxima de las secuencias
model = tr.create_model(state_dim, num_actions, max_seq_len)

# Entrenar el modelo
tr.train_model(model, X_train, Y_train, X_val, Y_val, max_seq_len)

# Validar el modelo
tr.validate_model(model, X_val, Y_val)