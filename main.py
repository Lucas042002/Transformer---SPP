import hr_algorithm as hr
import transformer as tr
import hr_transformer as hr_tr
import hr_random as hr_rand
import generator as gen
import torch
import torch.nn as nn
import numpy as np
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
cantidad = 1    # Cambia aquí la cantidad de problemas a generar
exportar = True  # Cambia a True si quieres guardar los archivos
# problemas, ancho, alto = gen.generate_problems_guillotine(categoria, cantidad, export=exportar)

problemas = gen.generate_problems_from_file("tests/c1p1.txt")
ancho, alto = CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"]
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
    all_states_total.extend(all_states)
    all_Y_rect_total.extend(all_Y_rect)
    # print(f"Altura final: {altura}")
    # print(f"Mejor altura final: {altura}")
    hr.visualizar_packing(placements, container_width=ancho, container_height=alto, show=True)


# Imprimir solo los "R_in" de cada estado en all_states_total
for state in all_states_total:
    print(f"largo: {len(state)}")
    for x in state:
        print(f"R_select_mask: {x.get('R_select_mask', [])}")

for fila in all_Y_rect_total:
    print(f"largo: {len(fila)}")
    for y in fila:
        print(f"y: {y}")

# Mostrar estados y los índices de los rectángulos elegidos si están disponibles
# if all_states_total is not None:
#     print(f"\nCantidad de secuencias de estados generadas: {len(all_states_total), len(all_Y_rect_total)}")

# largo_max = CATEGORIES[categoria]["num_items"] + 1

# train_loader, val_loader, input_seq_length, output_seq_length = tr.procesar_datos_entrada(
#     largo_max, all_states_total, all_Y_rect_total, verbose=True)

# Modelo
# input_dim = 5
# num_layers = 8 
# num_heads = 4
# head_dim = 8
# num_classes = largo_max  # O el número real de posibles acciones por paso
# learning_rate = 1e-4  # o incluso 1e-5

# model = tr.CustomModel(input_dim=input_dim, num_heads=num_heads, head_dim=head_dim, num_layers=num_layers, num_classes=num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Y = []
# for estados, acciones in zip(all_states_total, all_Y_rect_total):
#     for state, action in zip(estados, acciones):
#         if isinstance(action, list) or isinstance(action, np.ndarray):
#             if np.sum(action) == 0:
#                 Y.append(0)
#             else:
#                 Y.append(int(np.argmax(action)))
#         else:
#             Y.append(int(action))
# Y = np.array(Y)

# # Calcula los pesos inversos a la frecuencia
# num_classes = largo_max
# counts = np.bincount(Y, minlength=num_classes)
# weights = 1.0 / (counts + 1e-6)
# weights = weights / weights.sum() * num_classes  # Normaliza para que sumen num_classes
# weights = torch.tensor(weights, dtype=torch.float32)

# # print("Pesos para CrossEntropyLoss:", weights)

# criterion = nn.CrossEntropyLoss(weight=weights)




# # tr.entrenamiento(model, train_loader, val_loader, optimizer, criterion, epochs=50, categoria=categoria)

# # Cargar el modelo entrenado


# model_path = f"SPP_transformer_{categoria}.pth"  # Cambia por tu ruta real
# model.load_state_dict(torch.load(model_path, map_location="cpu"))  # Usa "cuda" si tienes GPU
# model.eval()




def limpiar_datos(all_states, all_Y_rect, largo_max):
    # Si solo tienes una instancia (una lista de estados y una de acciones)
    estados = all_states
    acciones = all_Y_rect

    for j, state in enumerate(estados):
        if len(state) < largo_max:
            # Rellenar con [0,0,0,0,0] hasta alcanzar largo_max
            state += [[0, 0, 0, 0, 0]] * (largo_max - len(state))


    # Filtrar estados y acciones donde la acción es solo ceros
    X = []
    Y = []
    for estado, accion in zip(estados, acciones):
        if not (isinstance(accion, list) and all(a == 0 for a in accion)):
            X.append(estado)
            # Convierte one-hot a índice (o 0 si todo es cero)
            if isinstance(accion, list) or isinstance(accion, np.ndarray):
                if np.sum(accion) == 0:
                    Y.append(0)
                else:
                    Y.append(int(np.argmax(accion)))
            else:
                Y.append(int(accion))

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def evaluar_modelo_vs_hr(problemas, ancho, alto, model, device="cpu"):
    total_pasos = 0
    pasos_correctos = 0
    cross_entropy_total = 0
    total_secuencias = 0
    secuencias_exactas = 0
    alturas_hr = []
    alturas_modelo = []
    diferencias_altura = []

    alturas_random = []
    diferencias_altura_random = []

    for idx, rects in enumerate(problemas):
        # Ejecuta HR
        placements_hr, altura_hr, secuencia_hr, estados_hr, Y_rect_hr, best_placement_states, best_placement_Y_states = hr.heuristic_recursion(rects, ancho)
        alturas_hr.append(altura_hr)

        # Ejecuta modelo
        placements_modelo, estados_modelo, Y_rect_modelo, acciones_modelo, logits_modelo = hr_tr.hr_packing_con_modelo(
            spaces=[(0, 0, ancho, 1000)],
            rects=rects.copy(),
            model=model,
            device=device
        )
        altura_modelo = hr_tr.calcular_altura(placements_modelo)
        alturas_modelo.append(altura_modelo)
        diferencias_altura.append(altura_modelo - altura_hr)


        # Ejecuta HR_random
        placements_random, estados_random, Y_rect_random = hr_rand.hr_packing_random(
            spaces=[(0, 0, ancho, 1000)],
            rects=rects.copy() 
        )
        altura_random = hr_tr.calcular_altura(placements_random)
        alturas_random.append(altura_random)
        diferencias_altura_random.append(altura_random - altura_modelo)


        # print(f" best_placement_Y_states: {(best_placement_Y_states)}")
        # print(f" Y_rect_modelo: {(Y_rect_modelo)}")

        # Precisión de decisión y cross entropy
        best_placement_states, best_placement_Y_states = limpiar_datos(
             best_placement_states, best_placement_Y_states, largo_max=CATEGORIES[categoria]["num_items"] + 1
        )
        # print(f" best_placement_Y_states: {(best_placement_Y_states)}")
        # print(f" acciones_modelo: {(acciones_modelo)}")



        pasos = min(len(best_placement_Y_states), len(acciones_modelo))
        correctos = 0
        ce = 0
        for t in range(pasos):
            # best_placement_Y_states[t] es one-hot o índice, acciones_modelo[t] es índice
            y_val = best_placement_Y_states[t]
            # Si es lista de listas, toma la primera sublista no vacía
            if isinstance(y_val, (list, np.ndarray)):
                # Si es lista de listas (ej: [[0,1,0,0], [0,0,0,0], ...])
                if len(y_val) > 0 and isinstance(y_val[0], (list, np.ndarray)):
                    # Busca la primera sublista que no sea todo ceros
                    for sub in y_val:
                        if np.sum(sub) > 0:
                            y_true = int(np.argmax(sub))
                            break
                    else:
                        y_true = 0  # Si todas son ceros, padding
                else:
                    # Es un one-hot plano
                    y_true = int(np.argmax(y_val))
            else:
                y_true = int(y_val)
            
            
            y_pred = int(acciones_modelo[t])
            if y_true == y_pred:
                correctos += 1

            # Cross entropy (usando logits si los guardas, si no, puedes omitir)
            # logits_modelo[t] = ... # Si guardas los logits en hr_packing_con_modelo
            ce += F.cross_entropy(torch.tensor([logits_modelo[t]]), torch.tensor([y_true]), reduction='sum').item()

        total_pasos += pasos
        pasos_correctos += correctos
        cross_entropy_total += ce

        total_secuencias += 1
        if correctos == pasos:
            secuencias_exactas += 1

        print(f"Problema {idx+1}: Altura HR={altura_hr}, Altura Modelo={altura_modelo}, Precisión de decisión={correctos/pasos:.3f}")


        # Guardar la mejor y peor imagen del modelo
        # Mejor: menor altura_modelo - altura_hr (más cercano a 0 o negativo)
        # Peor: mayor diferencia positiva (modelo peor que HR)
        if idx == 0:
            mejor_idx = idx
            peor_idx = idx
            mejor_diff = altura_modelo - altura_hr
            peor_diff = altura_modelo - altura_hr
            mejor_placements = placements_modelo
            peor_placements = placements_modelo
        else:
            diff = altura_modelo - altura_hr
            if diff < mejor_diff:
                mejor_diff = diff
                mejor_idx = idx
                mejor_placements = placements_modelo
            if diff > peor_diff:
                peor_diff = diff
                peor_idx = idx
                peor_placements = placements_modelo

    # Guardar imágenes después del bucle
    import matplotlib.pyplot as plt

    def guardar_packing(placements, ancho, alto, filename):
        fig = hr.visualizar_packing(placements, ancho, alto, show=False)
        plt.savefig(filename)
        plt.close(fig)

    mejor_filename = f"mejor_modelo_{categoria}.png"
    peor_filename = f"peor_modelo_{categoria}.png"
    guardar_packing(mejor_placements, CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"], mejor_filename)
    guardar_packing(peor_placements, CATEGORIES[categoria]["width"], CATEGORIES[categoria]["height"], peor_filename)

    print("\n--- MÉTRICAS GLOBALES ---")
    print(f"Precisión de decisión global: {pasos_correctos/total_pasos:.4f}")
    print(f"Cross Entropy promedio: {cross_entropy_total/total_pasos:.4f}")
    print(f"Altura promedio HR: {np.mean(alturas_hr):.2f}")
    print(f"Altura promedio Modelo: {np.mean(alturas_modelo):.2f}")
    std_dif = np.std(diferencias_altura)
    print(f"Diferencia promedio de altura: {np.mean(diferencias_altura):.2f} ± {std_dif:.2f}")
    print(f"Altura promedio Random: {np.mean(alturas_random):.2f}")
    std_dif_random = np.std(diferencias_altura_random)
    print(f"Diferencia promedio de altura (Random): {np.mean(diferencias_altura_random):.2f} ± {std_dif_random:.2f}")
    print(f"Exactitud de secuencia: {secuencias_exactas}/{total_secuencias} ({secuencias_exactas/total_secuencias:.2%})")




# --- Llama a la función de evaluación ---
# evaluar_modelo_vs_hr(
#     problemas=problemas,
#     ancho=ancho,
#     alto=alto,
#     model=model,
#     device="cpu"
# )