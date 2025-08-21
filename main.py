import hr_algorithm as hr
import transformer_encoder_decoder as tr
import hr_transformer as hr_tr
import hr_random as hr_rand
import generator as gen
import torch
import torch.nn as nn
import numpy as np
import pprint
import states as st
import torch.nn.functional as F

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



# ----------------------------
# Elegir categoría y cantidad de problemas
categoria = "C1"  # Cambia aquí la categoría
cantidad = 30    # Cambia aquí la cantidad de problemas a generar
exportar = False  # Cambia a True si quieres guardar los archivos
problemas, ancho, alto = gen.generate_problems_guillotine(categoria, cantidad, export=exportar)
max_len = CATEGORIES[categoria]["num_items"] + 1  # +1 para seq_id


# problemas = gen.generate_problems_from_file("tests/c1p1.txt")
# ancho, alto = CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"]
# max_len = CATEGORIES[categoria]["num_items"]  # +1 para seq_id

# print(f"Problemas generados para la categoría {categoria}: {problemas[0]}")



# ----------------------------
# Ejecutar
# ----------------------------

all_states_total = []
all_Y_rect_total = []


#Ahora puedes pasar cada problema al HR y recolectar los datos
for idx, rects in enumerate(problemas):
    # print(f"\nResolviendo problema {idx+1} de la categoría {categoria} ({len(rects)} rectángulos, contenedor {ancho}x{alto})")
    placements, altura, rect_sequence, all_states, all_Y_rect, best_placement_states, best_placement_Y_states = hr.heuristic_recursion(rects, ancho, category=categoria)

    aux = []
    for state in all_states:
        aux = st.agregar_seq_id_estados(state)
        all_states_total.append(aux)

    all_Y_rect_total.extend(all_Y_rect)
    # print(f"Altura final: {altura}")
    # print(f"Mejor altura final: {altura}")
    # hr.visualizar_packing(placements, container_width=ancho, container_height=alto, show=True)

all_states_total_new = []
for instance in all_states_total:
    for state in instance:
        new_state = []
        for space in state[0]:
            new_state.append(space)
        for rect in state[1]:
            new_state.append(rect)
        if len(new_state) < max_len:
            new_state += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, state[0][0][12]]] * (max_len - len(new_state))
        all_states_total_new.append(new_state)


all_Y_rect_total_new = []
for instance in all_Y_rect_total:
    for desicion in instance:
        all_Y_rect_total_new.append(desicion)


X_tensor = torch.tensor(all_states_total_new, dtype=torch.float32)
Y_tensor = torch.tensor(all_Y_rect_total_new, dtype=torch.long)

# Después de crear X_tensor y Y_tensor
print("Shape x:", X_tensor.shape)  # (148, 17, 13)
print("Shape y:", Y_tensor.shape)  # (148,)

# Usar la función adaptada
train_loader, val_loader, input_seq_length, output_seq_length = tr.procesar_datos_entrada_encoder_decoder_adapted(
    X_tensor, Y_tensor, verbose=True
)

# Crear el modelo
input_dim = 13  # Dimensión de los vectores de estado
num_classes = 18  # 17 rectángulos + 1 token especial (start/end)

model = tr.SPPTransformerEncoderDecoder(
    input_dim=input_dim,
    d_model=512,
    num_heads=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    d_ff=512,
    max_seq_length=input_seq_length,  # 17
    num_classes=num_classes,  # 18
    dropout=0.1
)

# Entrenar el modelo
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora padding

train_losses, val_accuracies = tr.entrenamiento_encoder_decoder(
    model, train_loader, val_loader, optimizer, criterion, 
    epochs=50, categoria=categoria
)

print("Entrenamiento completado!")
