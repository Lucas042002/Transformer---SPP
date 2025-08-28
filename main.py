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
import categories as cat

# ----------------------------
# Datos del problema
# ----------------------------

# ----------------------------
# Elegir categoría y cantidad de problemas
categoria = "C1"  # Cambia aquí la categoría
cantidad = 100    # Cambia aquí la cantidad de problemas a generar
exportar = False  # Cambia a True si quieres guardar los archivos
problemas, ancho, alto = gen.generate_problems_guillotine(categoria, cantidad, export=exportar)
max_len = cat.CATEGORIES[categoria]["num_items"] + 1  # +1 para seq_id


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

    # aux = []
    for problem in all_states:
        
        all_states_total.append(problem)
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
                        new_state += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (max_len - len(new_state))
                    
                    all_states_total_new.append(new_state)
                    all_Y_rect_total_new.append(decision)

# print(f"Estados válidos después del filtrado (sin decisiones 0): {len(all_states_total_new)}")
# print(f"Decisiones válidas después del filtrado (sin decisiones 0): {len(all_Y_rect_total_new)}")

# Verificar que no hay decisiones 0 en los datos finales
# if len(all_Y_rect_total_new) > 0:
    # print(f"Rango de decisiones: {min(all_Y_rect_total_new)} - {max(all_Y_rect_total_new)}")
    # print(f"Número de decisiones únicas: {len(set(all_Y_rect_total_new))}")
    # print(f"Distribución de decisiones: {np.bincount(all_Y_rect_total_new)}")


X_tensor = torch.tensor(all_states_total_new, dtype=torch.float32)
Y_tensor = torch.tensor(all_Y_rect_total_new, dtype=torch.long)
# Después de crear X_tensor y Y_tensor
print("Shape x:", X_tensor.shape)  # Esperado: (N, 18, 12) - sin seq_id
print("Shape y:", Y_tensor.shape)  # Esperado: (N,)

# Usar la función adaptada
train_loader, val_loader, input_seq_length, output_seq_length = tr.procesar_datos_entrada_encoder_decoder_adapted(
    X_tensor, Y_tensor, verbose=False
)


# Crear el modelo
model = tr.SPPTransformer(
    d_model=256,
    num_heads=8,
    d_ff=512,
    num_layers=6,
    input_dim=12,  # 12 features sin seq_id
    num_classes=18,  # 17 rectángulos + tokens especiales
    max_len=max_len+2,  # Usar max_len definido arriba
    dropout=0.1
).to('cpu')
# Usar tus datos actuales (ya procesados)
train_loader, val_loader, _, _ = tr.procesar_datos_entrada_encoder_decoder_adapted(
    X_tensor, Y_tensor, verbose=True
)
#Entrenar
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#criterion = nn.CrossEntropyLoss(ignore_index=0)
#train_losses, val_accuracies = tr.entrenar_spp_transformer(
#    model, train_loader, val_loader, optimizer, criterion, epochs=50
#)
#model = tr.SPPTransformer(
#    d_model=256,
#    num_heads=8,
#    d_ff=512,
#    num_layers=6,
#    input_dim=12,  # Tus 12 features de estado
#    num_classes=18,  # 17 rectángulos + tokens especiales
#    max_len=max_len+2,  # Usar max_len definido arriba
#    dropout=0.1
#).to('cpu')
#Cargar modelo entrenado
checkpoint = tr.cargar_modelo(model, "models/spp_transformer_c1_epochs50.pth")

# ----------------------------
# Probar el modelo con HR_Transformer
# ----------------------------

# Generar problemas de prueba
problemas_test, ancho_test, alto_test = gen.generate_problems_guillotine(categoria, 1, export=False)
print(f"Problema de prueba generado: {problemas_test[0]}")


# Antes de probar el modelo
print(f"Forma del estado de ejemplo: {len(all_states_total_new[0])}")
print(f"Forma de una fila del estado: {len(all_states_total_new[0][0])}")
print(f"Tipo de modelo: {type(model).__name__}")

# Verificar que el modelo puede procesar los datos
with torch.no_grad():
    test_input = torch.tensor([all_states_total_new[0]], dtype=torch.float32)
    print(f"Tensor de prueba shape: {test_input.shape}")
    try:
        # Para modelo encoder-decoder, necesitas ambos inputs:
        decoder_input = torch.tensor([[9]], dtype=torch.long)  # Start token
        test_output = model(test_input, decoder_input)
        print(f"Salida del modelo shape: {test_output.shape}")
        print("✓ Modelo funciona correctamente")
    except Exception as e:
        print(f"Error en modelo: {e}")


# Función helper para usar el modelo en inferencia
#def predecir_con_modelo(model, estado_codificado, device='cpu'):
#    """
#    Usa el modelo encoder-decoder para predecir la próxima acción
#    """
#    model.eval()
#     with torch.no_grad():
#         # Encoder input: estado actual
#         encoder_input = torch.tensor([estado_codificado], dtype=torch.float32).to(device)
        
#         # Decoder input: solo start token
#         decoder_input = torch.tensor([[9]], dtype=torch.long).to(device)  # 9 = start token
        
#         # Forward pass
#         logits = model(encoder_input, decoder_input)  # [1, seq_len, num_classes]
#         logits = logits[:, -1, :]  # Tomar última predicción [1, num_classes]
        
#         # Obtener probabilidades y predicción
#         probs = torch.softmax(logits, dim=-1)
#         prediction = torch.argmax(logits, dim=-1).item()
        
#         return prediction, probs.cpu().numpy().flatten()

# Usar el modelo entrenado con HR_Transformer
#print(f"\n--- Probando modelo con HR_Transformer ---")
for idx, rects_test in enumerate(problemas_test):
    #print(f"\nResolviendo problema de prueba {idx+1} con modelo entrenado:")
    #print(f"Rectángulos: {rects_test}")

    # Llamar a HR_Transformer con el modelo cargado
    placements_modelo, altura_modelo, rect_sequence_modelo, all_states_modelo, all_Y_rect_modelo = hr_tr.heuristic_recursion_transformer(
        rects=rects_test.copy(),
        container_width=ancho_test,
        model=model,
        device='cpu',
        category=categoria
    )

    print(f"Altura conseguida con modelo: {altura_modelo}")
    print(f"Secuencia de rectángulos: {rect_sequence_modelo}")
    print(f"Número de estados explorados: {len(all_states_modelo)}")
    # Comparar con HR original
    print(f"\n--- Comparación con HR original ---")
    placements_hr, altura_hr, rect_sequence_hr, all_states_hr, all_Y_rect_hr, _, _ = hr.heuristic_recursion(
        rects=rects_test.copy(), 
        container_width=ancho_test, 
        category=categoria
    )
    print(f"Altura HR original: {altura_hr}")
    print(f"Altura con modelo: {altura_modelo}")
    print(f"Mejora: {((altura_hr - altura_modelo) / altura_hr * 100):.2f}%" if altura_hr > 0 else "N/A")

# ----------------------------
# Probar predicciones individuales
# ----------------------------
# print(f"\n--- Probando predicciones individuales ---")

# # Tomar un estado de ejemplo
# estado_ejemplo = all_states_total_new[0]
# decision_real = all_Y_rect_total_new[0]

# prediction, probs = predecir_con_modelo(model, estado_ejemplo)

# print(f"Estado de ejemplo shape: {np.array(estado_ejemplo).shape}")
# print(f"Decisión real: {decision_real}")
# print(f"Predicción del modelo: {prediction}")
# print(f"Probabilidades top-5:")
# top_indices = np.argsort(probs)[-5:][::-1]
# for i, idx in enumerate(top_indices):
#     print(f"  {i+1}. Clase {idx}: {probs[idx]:.4f}")

# # Verificar accuracy en algunos ejemplos
# print(f"\n--- Accuracy en muestra de validación ---")
# correct = 0
# total = 0
# model.eval()

# with torch.no_grad():
#     for i in range(min(10, len(all_states_total_new))):
#         estado = all_states_total_new[i]
#         decision_real = all_Y_rect_total_new[i]
        
#         prediction, _ = predecir_con_modelo(model, estado)
        
#         if prediction == decision_real:
#             correct += 1
#         total += 1
        
#         print(f"Ejemplo {i+1}: Real={decision_real}, Pred={prediction}, {'✓' if prediction == decision_real else '✗'}")

# print(f"\nAccuracy en muestra: {correct}/{total} = {correct/total:.2%}")