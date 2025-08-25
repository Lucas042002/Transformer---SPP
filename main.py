import hr_algorithm as hr
import transformer_attention_original as tr
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
categoria = "C4"  # Cambia aquí la categoría
cantidad = 1    # Cambia aquí la cantidad de problemas a generar
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
    for problem in all_states:
        aux = st.agregar_seq_id_estados(problem)
        all_states_total.append(aux)

    all_Y_rect_total.extend(all_Y_rect)
    # print(f"Altura final: {altura}")
    # print(f"Mejor altura final: {altura}")
    # hr.visualizar_packing(placements, container_width=ancho, container_height=alto, show=True)

# ...existing code...

all_states_total_new = []
all_Y_rect_total_new = []

for instance_idx, instance in enumerate(all_states_total):
    for state_idx, state in enumerate(instance):
        # Verificar si hay rectángulos disponibles para seleccionar
        rectangulos_disponibles = state[1]  # Los rectángulos están en state[1]
        
        # Solo procesar estados donde hay rectángulos disponibles
        if len(rectangulos_disponibles) > 0:
            # Verificar que existe una decisión correspondiente
            if state_idx < len(all_Y_rect_total[instance_idx]):
                decision = all_Y_rect_total[instance_idx][state_idx]
                
                # Verificar que la decisión sea válida (no -1, no 0)
                if decision > 0:  # ← Cambio aquí: decision > 0 en lugar de >= 0
                    new_state = []
                    # Agregar espacios
                    for space in state[0]:
                        new_state.append(space)
                    # Agregar rectángulos
                    for rect in state[1]:
                        new_state.append(rect)
                    
                    # Padding si es necesario
                    if len(new_state) < max_len:
                        new_state += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, state[0][0][12]]] * (max_len - len(new_state))
                    
                    all_states_total_new.append(new_state)
                    all_Y_rect_total_new.append(decision)

print(f"Estados válidos después del filtrado (sin decisiones 0): {len(all_states_total_new)}")
print(f"Decisiones válidas después del filtrado (sin decisiones 0): {len(all_Y_rect_total_new)}")

# Verificar que no hay decisiones 0 en los datos finales
if len(all_Y_rect_total_new) > 0:
    print(f"Rango de decisiones: {min(all_Y_rect_total_new)} - {max(all_Y_rect_total_new)}")
    print(f"Número de decisiones únicas: {len(set(all_Y_rect_total_new))}")
    print(f"Distribución de decisiones: {np.bincount(all_Y_rect_total_new)}")


X_tensor = torch.tensor(all_states_total_new, dtype=torch.float32)
Y_tensor = torch.tensor(all_Y_rect_total_new, dtype=torch.long)
# Después de crear X_tensor y Y_tensor
print("Shape x:", X_tensor.shape)  # (148, 18, 13)
print("Shape y:", Y_tensor.shape)  # (148,)

# # Usar la función adaptada
# train_loader, val_loader, input_seq_length, output_seq_length = tr.procesar_datos_entrada_encoder_decoder_adapted(
#     X_tensor, Y_tensor, verbose=True
# )
# # Crear el modelo
# model = tr.SPPTransformer(
#     d_model=256,
#     num_heads=8,
#     d_ff=512,
#     num_layers=6,
#     input_dim=13,  # Tus 13 features de estado
#     num_classes=18,  # 17 rectángulos + tokens especiales
#     max_len=max_len+2,  # Usar max_len definido arriba
#     dropout=0.1
# ).to('cpu')
# # Usar tus datos actuales (ya procesados)
# train_loader, val_loader, _, _ = tr.procesar_datos_entrada_encoder_decoder_adapted(
#     X_tensor, Y_tensor, verbose=True
# )
# # Entrenar
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# train_losses, val_accuracies = tr.entrenar_spp_transformer(
#     model, train_loader, val_loader, optimizer, criterion, epochs=50
# )
# print("Entrenamiento completado!")



# # Cargar modelo entrenado
# checkpoint = tr.cargar_modelo(model, "models/spp_transformer_c1_epochs50.pth")

# # Usar modelo cargado para predicciones
# model.eval()
# with torch.no_grad():
#     predictions = model(test_data)