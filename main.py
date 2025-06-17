import hr_algorithm as hr
import transformer as tr
import hr_transformer as hr_tr
import torch
import torch.nn as nn
import random
import numpy as np
# ----------------------------
# Datos del problema
# ----------------------------



CATEGORIES = {
    "C1": {"num_items": 17, "width": 20, "height": 20},
    "C2": {"num_items": 25, "width": 40, "height": 15},
    "C3": {"num_items": 29, "width": 60, "height": 30},
    "C4": {"num_items": 49, "width": 60, "height": 60},
    "C5": {"num_items": 73, "width": 60, "height": 90},
    "C6": {"num_items": 97, "width": 80, "height": 120},
    "C7": {"num_items": 197, "width": 160, "height": 240},
}

def random_rectangle_discrete(max_width, max_height, min_aspect=1, max_aspect=3, allowed_sizes=None):
    if allowed_sizes is None:
        allowed_sizes = list(range(2, max(max_width, max_height)+1))
    while True:
        w = random.choice(allowed_sizes)
        h = random.choice(allowed_sizes)
        aspect = max(w/h, h/w)
        if w <= max_width and h <= max_height and min_aspect <= aspect <= max_aspect:
            return (w, h)

def generate_problems(category, n_problems=3, export=False):
    if category not in CATEGORIES:
        raise ValueError(f"Categoría no válida. Opciones: {list(CATEGORIES.keys())}")
    cat = CATEGORIES[category]
    problems = []
    allowed_sizes = list(range(2, min(cat["width"], cat["height"])+1))
    target_area = cat["width"] * cat["height"]
    num_rects = cat["num_items"]
    area_prom = target_area // num_rects

    for i in range(n_problems):
        intentos = 0
        while True:
            rects = []
            total_area = 0
            # Genera n-1 rectángulos con área cercana al promedio
            for _ in range(num_rects - 1):
                for _ in range(100):  # 100 intentos por rectángulo
                    w = random.choice(allowed_sizes)
                    h = random.choice(allowed_sizes)
                    aspect = max(w/h, h/w)
                    area = w * h
                    if (
                        w <= cat["width"] and h <= cat["height"] and
                        1 <= aspect <= 3 and
                        abs(area - area_prom) <= area_prom // 2  # área cercana al promedio
                    ):
                        rects.append((w, h))
                        total_area += area
                        break
            area_faltante = target_area - total_area
            # Busca un rectángulo válido para completar el área exacta
            encontrado = False
            for w in allowed_sizes:
                if area_faltante % w != 0:
                    continue
                h = area_faltante // w
                aspect = max(w/h, h/w) if h > 0 else 999
                if (
                    h in allowed_sizes and
                    w <= cat["width"] and h <= cat["height"] and
                    1 <= aspect <= 3
                ):
                    rects.append((w, h))
                    encontrado = True
                    break
            if encontrado:
                problems.append(rects)
                if export:
                    fname = f"{category.lower()}_random_{i+1}.txt"
                    with open(fname, "w") as f:
                        for w, h in rects:
                            f.write(f"{h}\t{w}\n")
                    print(f"Exportado: {fname} (Área total: {sum(w*h for w,h in rects)})")
                break
            intentos += 1
            if intentos > 1000:
                raise RuntimeError("No se pudo generar un problema válido tras muchos intentos.")
    return problems, cat["width"], cat["height"]

def generate_problems_from_file(filepath):
    """
    Lee un archivo de problema (como c1p1.txt) y devuelve una lista de rectángulos [(w, h), ...].
    """
    rects = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, w = map(int, parts)
            rects.append((w, h))
    return [rects]  # Para mantener compatibilidad con el resto del flujo


# ----------------------------
# Elegir categoría y cantidad de problemas
categoria = "C1"  # Cambia aquí la categoría
cantidad = 4      # Cambia aquí la cantidad de problemas a generar
exportar = False  # Cambia a True si quieres guardar los archivos
problemas, ancho, alto = generate_problems(categoria, cantidad, export=exportar)

# problemas = generate_problems_from_file("c1p1.txt")
# ancho, alto = CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"]


# ----------------------------
# Ejecutar
# ----------------------------

all_states_total = []
all_Y_rect_total = []
# Ahora puedes pasar cada problema al HR y recolectar los datos
for idx, rects in enumerate(problemas):
    print(f"\nResolviendo problema {idx+1} de la categoría {categoria} ({len(rects)} rectángulos, contenedor {ancho}x{alto})")
    result = hr.heuristic_recursion(rects, ancho)
    placements, altura, rect_sequence, all_states, all_Y_rect = result
    all_states_total.extend(all_states)
    all_Y_rect_total.extend(all_Y_rect)
    print(f"Mejor altura final: {altura}")


# Visualizar el packing
# hr.visualizar_packing(placements, CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"])

# Mostrar estados y los índices de los rectángulos elegidos si están disponibles
if all_states_total is not None:
    print(f"\nCantidad de secuencias de estados generadas: {len(all_states_total), len(all_Y_rect_total)}")

largo_max = CATEGORIES[categoria]["num_items"] + 1

train_loader, val_loader, input_seq_length, output_seq_length = tr.procesar_datos_entrada(
    largo_max, all_states_total, all_Y_rect_total, verbose=False)

# Modelo
input_dim = 5
num_layers = 3
num_heads = 3
head_dim = 9
num_classes = largo_max  # O el número real de posibles acciones por paso

model = tr.CustomModel(input_dim=input_dim, num_heads=num_heads, head_dim=head_dim, num_layers=num_layers, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

Y = []
for estados, acciones in zip(all_states_total, all_Y_rect_total):
    for state, action in zip(estados, acciones):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            if np.sum(action) == 0:
                Y.append(0)
            else:
                Y.append(int(np.argmax(action)))
        else:
            Y.append(int(action))
Y = np.array(Y)

# Calcula los pesos inversos a la frecuencia
num_classes = largo_max
counts = np.bincount(Y, minlength=num_classes)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * num_classes  # Normaliza para que sumen num_classes
weights = torch.tensor(weights, dtype=torch.float32)

print("Pesos para CrossEntropyLoss:", weights)

criterion = nn.CrossEntropyLoss(weight=weights)



tr.entrenamiento(model, train_loader, val_loader, optimizer, criterion)

# model_path = "SPP_transfomer_insano.pth"  # Cambia por tu ruta real
# model.load_state_dict(torch.load(model_path, map_location="cpu"))  # Usa "cuda" si tienes GPU
# model.eval()
# print("Modelo cargado correctamente.")

# ----------------------------
# Probar el modelo con los 10 estados dados


# Lista de 10 secuencias de estados (cada una es una lista de listas)
test_states = [
    [ [5, 6, 30, 0, 0], [1000, 20, 20000, 1, 1], [4, 8, 32, 0, 0], [4, 7, 28, 0, 0],  [7, 4, 28, 0, 0],[9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0]],
    [[995, 20, 19900, 0, 1], [5, 14, 70, 1, 1], [4, 8, 32, 0, 0], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0]],
    [[995, 20, 19900, 0, 1], [5, 6, 30, 1, 1], [1, 8, 8, 0, 1], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0]],
    [[995, 20, 19900, 0, 1], [5, 1, 5, 1, 1], [0, 5, 0, 0, 1], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0]],
    [[995, 20, 19900, 0, 1], [0, 5, 0, 1, 1], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[995, 20, 19900, 0, 1], [1, 8, 8, 1, 1], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[995, 20, 19900, 1, 1], [7, 4, 28, 0, 0], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[988, 20, 19760, 0, 1], [7, 16, 112, 1, 1], [4, 7, 28, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[988, 20, 19760, 0, 1], [7, 9, 63, 1, 1], [3, 7, 21, 0, 1], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[988, 20, 19760, 0, 1], [7, 0, 0, 0, 1], [4, 9, 36, 1, 1], [9, 3, 27, 0, 0], [9, 3, 27, 0, 0], [5, 5, 25, 0, 0], [4, 6, 24, 0, 0], [3, 8, 24, 0, 0], [7, 3, 21, 0, 0], [3, 7, 21, 0, 0], [6, 3, 18, 0, 0], [4, 4, 16, 0, 0], [3, 5, 15, 0, 0], [6, 2, 12, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
]

# Convertir a tensores y pasar por el modelo

# model.eval()
# with torch.no_grad():
#     for i, seq in enumerate(test_states):
#         x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)
#         logits = model(x)  # (1, num_classes)
#         probs = torch.softmax(logits, dim=-1)  # (1, num_classes)
#         pred = torch.argmax(probs, dim=-1)     # (1,)
#         top3 = torch.topk(probs, 3, dim=-1)
#         print(f"Secuencia {i+1}:")
#         print(f"  Índice predicho: {pred.item()}")
#         print(f"  Probabilidades: {probs[0].cpu().numpy().round(3).tolist()}")
#         print(f"  Top 3 índices: {top3.indices[0].cpu().numpy().tolist()} (probs: {top3.values[0].cpu().numpy().round(10).tolist()})")
#         print()


for idx, rects in enumerate(problemas):
    print(f"\nResolviendo problema {idx+1} con el modelo")
    placements, estados, Y_rect = hr_tr.hr_packing_con_modelo(
        spaces=[(0, 0, ancho, 1000)],
        rects=rects.copy(),
        model=model,
        device="cpu"  # Cambia a "cuda" si tienes GPU disponible
    )
    print(f"Packing con modelo, altura final: {hr_tr.calcular_altura(placements)}")
    hr.visualizar_packing(placements, ancho, alto)
