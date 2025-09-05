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
cantidad = 1    # Cambia aquí la cantidad de problemas a generar
exportar = False  # Cambia a True si quieres guardar los archivos
problemas, ancho, alto = gen.generate_problems_guillotine(categoria, cantidad, export=exportar)
max_len_estado = cat.CATEGORIES[categoria]["num_items"] + 1  # Número máximo de items en un estado de ejecución

# problemas = gen.generate_problems_from_file("tests/c1p1.txt")
# ancho, alto = cat.CATEGORIES[categoria]["width"], cat.CATEGORIES[categoria]["height"]
# max_len_estado = CATEGORIES[categoria]["num_items"]  # Número máximo de items en un estado de ejecución

# print(f"Problemas generados para la categoría {categoria}: {problemas[0]}")


# ----------------------------
# Ejecutar
# ----------------------------

all_states_total = []
all_Y_rect_total = []

total_problemas = 0
#Ahora puedes pasar cada problema al HR y recolectar los datos
for idx, rects in enumerate(problemas):
    # print(f"\nResolviendo problema {idx+1} de la categoría {categoria} ({len(rects)} rectángulos, contenedor {ancho}x{alto})")
    placements, altura, rect_sequence, all_states, all_Y_rect, best_placement_states, best_placement_Y_states = hr.heuristic_recursion(rects, ancho, category=categoria)

    # aux = []
    for problem in all_states:
        total_problemas += 1
        all_states_total.append(problem)
    all_Y_rect_total.extend(all_Y_rect)
    # print(f"Altura final: {altura}")
    # print(f"Mejor altura final: {altura}")
    # hr.visualizar_packing(placements, container_width=ancho, container_height=alto, show=True)

# print("Total problemas: ", total_problemas)
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
                    # print(len(new_state))
                    # Padding si es necesario
                    if len(new_state) < max_len_estado:
                        new_state += [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * (max_len_estado - len(new_state))

                    all_states_total_new.append(new_state)
                    all_Y_rect_total_new.append(decision)

# for state in all_states_total_new:
#     print(state)
# for state in all_Y_rect_total_new:
#     print(state)
# Verificar que no hay decisiones 0 en los datos finales
# if len(all_Y_rect_total_new) > 0:
    # print(f"Rango de decisiones: {min(all_Y_rect_total_new)} - {max(all_Y_rect_total_new)}")
    # print(f"Número de decisiones únicas: {len(set(all_Y_rect_total_new))}")
    # print(f"Distribución de decisiones: {np.bincount(all_Y_rect_total_new)}")


X_tensor = torch.tensor(all_states_total_new, dtype=torch.float32)
Y_tensor = torch.tensor(all_Y_rect_total_new, dtype=torch.long)
# Después de crear X_tensor y Y_tensor
print("Shape x:", X_tensor.shape)  # Esperado: (N, categoria["num_items"] + 1, 12) 
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
    num_classes=max_len_estado + 2,  # 17 rectángulos + espacio inicial + tokens especiales
    max_len=max_len_estado,  # Usar num_items + 1 definido arriba
    dropout=0.1
).to('cpu')
# Entrenar
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# train_losses, val_accuracies = tr.entrenar_spp_transformer(
#     model, train_loader, val_loader, optimizer, criterion, epochs=50, categoria=categoria
# )

# Cargar modelo preentrenado
checkpoint = tr.cargar_modelo(model, "models/spp_transformer_c1_acc8897.pth")


# # Crear optimizer y cargar su estado
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # LR más bajo para fine-tuning
# if 'optimizer_state_dict' in checkpoint:
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# # Obtener épocas y métricas previas
# epoch_inicial = checkpoint['epoch']
# train_losses_previas = checkpoint['train_losses']
# val_accuracies_previas = checkpoint['val_accuracies']

# print(f"Continuando entrenamiento desde epoch {epoch_inicial}")
# print(f"Accuracy previa: {max(val_accuracies_previas):.4f}")

# # Reentrenar con más épocas
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# train_losses_nuevas, val_accuracies_nuevas = tr.entrenar_spp_transformer_continuo(
#     model, train_loader, val_loader, optimizer, criterion, 
#     epochs_adicionales=30,  # Épocas adicionales
#     train_losses_previas=train_losses_previas,
#     val_accuracies_previas=val_accuracies_previas,
#     epoch_inicial=epoch_inicial,
#     categoria=categoria
# )


# ----------------------------
# Probar el modelo con HR_Transformer
# ----------------------------

# # Generar problemas de prueba
# problemas_test, ancho_test, alto_test = gen.generate_problems_guillotine(categoria, 1, export=False)
# print(f"Problema de prueba generado: {problemas_test[0]}")


# # Antes de probar el modelo
# print(f"Forma del estado de ejemplo: {len(all_states_total_new[0])}")
# print(f"Forma de una fila del estado: {len(all_states_total_new[0][0])}")
# print(f"Tipo de modelo: {type(model).__name__}")

# Verificar que el modelo puede procesar los datos
with torch.no_grad():
    test_input = torch.tensor([all_states_total_new[0]], dtype=torch.float32)
    print(f"Tensor de prueba shape: {test_input.shape}")
    try:
        # Para modelo encoder-decoder, necesitas ambos inputs:
        decoder_input = torch.tensor([[9]], dtype=torch.long)  # Start token
        test_output = model(test_input, decoder_input)
        print(f"Salida del modelo shape: {test_output.shape}")
        print("Modelo funciona correctamente")
    except Exception as e:
        print(f"Error en modelo: {e}")


# # Usar el modelo entrenado con HR_Transformer
# print(f"\n--- Probando modelo con HR_Transformer ---")
# for idx, rects_test in enumerate(problemas_test):
#     print(f"\nResolviendo problema de prueba {idx+1} con modelo entrenado:")
#     print(f"Rectángulos: {rects_test}")
#     # Llamar a HR_Transformer con el modelo cargado
#     placements_modelo, altura_modelo, rect_sequence_modelo, all_states_modelo, all_Y_rect_modelo = hr_tr.heuristic_recursion_transformer(
#         rects=rects_test.copy(),
#         container_width=ancho_test,
#         model=model,
#         device='cpu',
#         category=categoria
#     )
#     print(f"Altura conseguida con modelo: {altura_modelo}")
#     print(f"Secuencia de rectángulos: {rect_sequence_modelo}")
#     print(f"Número de estados explorados: {len(all_states_modelo)}")
#     # Comparar con HR original
#     print(f"\n--- Comparación con HR original ---")
#     placements_hr, altura_hr, rect_sequence_hr, all_states_hr, all_Y_rect_hr, _, _ = hr.heuristic_recursion(
#          rects=rects_test.copy(), 
#          container_width=ancho_test, 
#          category=categoria
#     )
#     print(f"Altura HR original: {altura_hr}")
#     print(f"Altura con modelo: {altura_modelo}")
#     print(f"Diferencia con el HR original: {((altura_hr - altura_modelo) / altura_hr * 100):.2f}%" if altura_hr > 0 else "N/A")




def evaluar_modelo_vs_hr(problemas, ancho, alto, model, device="cpu", categoria="C1"):
    """
    Evalúa el modelo transformer vs HR original con métricas completas
    """
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
    
    print(f"\n=== EVALUANDO MODELO VS HR EN {len(problemas)} PROBLEMAS ===")
    
    for idx, rects in enumerate(problemas):
        print(f"\nProcesando problema {idx+1}/{len(problemas)}...")
        
        # 1. Ejecutar HR original
        try:
            placements_hr, altura_hr, secuencia_hr, estados_hr, Y_rect_hr, best_placement_states, best_placement_Y_states = hr.heuristic_recursion(
                rects.copy(), ancho, category=categoria
            )
            alturas_hr.append(altura_hr)
        except Exception as e:
            print(f"Error en HR original: {e}")
            continue
        
        # print(f"Altura HR original: {altura_hr}")
        # print(f"Mejores estados de colocación (HR): {len(best_placement_states)}")
        # print(f"Mejores decisiones (HR): {len(best_placement_Y_states)}")


        # 2. Ejecutar modelo transformer
        try:
            placements_modelo, altura_modelo, rect_sequence_modelo, all_states_modelo, all_Y_rect_modelo = hr_tr.heuristic_recursion_transformer(
                rects=rects.copy(),
                container_width=ancho,
                model=model,
                device=device,
            category=categoria
            )
            alturas_modelo.append(altura_modelo)
            diferencias_altura.append(altura_modelo - altura_hr)
        except Exception as e:
            print(f"Error en modelo: {e}")
            alturas_modelo.append(float('inf'))
            diferencias_altura.append(float('inf'))
            continue

        try:
            placements_random, estados_random, Y_rect_random = hr_rand.hr_packing_random(
                spaces=[(0, 0, ancho, 1000)],
                rects=rects.copy()
            )
            altura_random = hr.calcular_altura(placements_random)
            alturas_random.append(altura_random)
            diferencias_altura_random.append(altura_random - altura_modelo)
        except Exception as e:
            print(f"Error en HR random: {e}")
            alturas_random.append(float('inf'))
            diferencias_altura_random.append(float('inf'))
        


        print(f"\n--- problema ---")
        print(f"Y HR: {best_placement_Y_states}\n\n")
        print(f"Y Modelo: {all_Y_rect_modelo}\n\n")
        print(f"Y Random: {Y_rect_random}\n\n")


# En main.py, líneas 290-320, reemplazar con:

        # 4. Calcular métricas de precisión de decisión
        try:            
            # Procesar estados de HR para comparación
            estados_hr_procesados = []
            decisiones_hr_procesadas = []
            
            largo_max = cat.CATEGORIES[categoria]["num_items"] + 1
            
                        
            # Procesar cada secuencia de estados de HR
            for seq_idx, secuencia_estados in enumerate(best_placement_states):
                
                if seq_idx < len(best_placement_Y_states):
                    decisiones_secuencia = best_placement_Y_states[seq_idx]
                    
                    # CORRECCIÓN: Normalizar decisiones_secuencia si es un entero
                    if isinstance(decisiones_secuencia, (int, np.integer)):
                        print(f"    DEBUG - decisiones_secuencia es entero: {decisiones_secuencia}, convirtiendo a lista")
                        decisiones_secuencia = [decisiones_secuencia]
                    
                    
                    for estado_idx, estado in enumerate(secuencia_estados):
                        
                        if estado_idx < len(decisiones_secuencia):
                            decision = decisiones_secuencia[estado_idx]
                            
                            # Procesar estado para comparación
                            estado_procesado = []
                            
                            
                            # Agregar espacios (verificar que estado[0] existe y es iterable)
                            if hasattr(estado[0], '__iter__'):
                                for space in estado[0]:
                                    estado_procesado.append(space)
                            
                            # Agregar rectángulos (verificar que estado[1] existe y es iterable)
                            if len(estado) > 1 and hasattr(estado[1], '__iter__'):
                                for rect in estado[1]:
                                    estado_procesado.append(rect)
                            else:
                                print(f"      WARNING: estado[1] no es iterable: {type(estado[1]) if len(estado) > 1 else 'no existe'}")
                            
                            # Padding
                            while len(estado_procesado) < largo_max:
                                estado_procesado.append([0] * 12)
                            
                            estados_hr_procesados.append(estado_procesado)
                            
                            # Procesar decisión
                            if isinstance(decision, (list, np.ndarray)):
                                if np.sum(decision) > 0:
                                    decision_idx = int(np.argmax(decision))
                                else:
                                    decision_idx = 0
                            else:
                                decision_idx = int(decision) if decision > 0 else 0
                            
                            decisiones_hr_procesadas.append(decision_idx)
            
            if len(all_states_modelo) > 0 and len(all_Y_rect_modelo) > 0:
                decisiones_modelo = all_Y_rect_modelo[0] if isinstance(all_Y_rect_modelo[0], list) else all_Y_rect_modelo
            else:
                decisiones_modelo = []
            
            print(f"  DEBUG - decisiones_modelo: {decisiones_modelo}")
            
            # Comparar decisiones paso a paso
            pasos_problema = min(len(decisiones_hr_procesadas), len(decisiones_modelo))
            correctos_problema = 0
            ce_problema = 0
            
            print(f"  DEBUG - Comparando {pasos_problema} pasos")
            
            for t in range(pasos_problema):
                y_true = decisiones_hr_procesadas[t]
                y_pred = decisiones_modelo[t] if t < len(decisiones_modelo) else 0
                
                if y_true == y_pred:
                    correctos_problema += 1
                
                # Cross entropy (simplificado)
                num_classes = cat.CATEGORIES[categoria]["num_items"] + 2
                logits_aprox = torch.ones(num_classes) * 0.1
                logits_aprox[y_pred] = 0.9  # Dar mayor probabilidad a la predicción
                
                try:
                    ce_step = F.cross_entropy(
                        logits_aprox.unsqueeze(0), 
                        torch.tensor([y_true]), 
                        reduction='sum'
                    ).item()
                    ce_problema += ce_step
                except:
                    ce_problema += 1.0  # Valor por defecto en caso de error
            
            total_pasos += pasos_problema
            pasos_correctos += correctos_problema
            cross_entropy_total += ce_problema
            total_secuencias += 1
            
            # Secuencia exacta si todas las decisiones son correctas
            if correctos_problema == pasos_problema and pasos_problema > 0:
                secuencias_exactas += 1
            
            precision_problema = correctos_problema / pasos_problema if pasos_problema > 0 else 0
            
        except Exception as e:
            print(f"Error calculando métricas de precisión: {e}")
            print(f"  DEBUG - Error en línea: {e.__traceback__.tb_lineno}")
            import traceback
            traceback.print_exc()
            total_secuencias += 1
            precision_problema = 0
        
        # Mostrar progreso
        mejora_porcentual = ((altura_hr - altura_modelo) / altura_hr * 100) if altura_hr > 0 else 0
        print(f"Problema {idx+1}: Altura HR={altura_hr:.1f}, Altura Modelo={altura_modelo:.1f}, "
              f"Mejora={mejora_porcentual:.1f}%, Precisión={precision_problema:.3f}")
    


    # 5. Calcular y mostrar métricas globales
    print("\n" + "="*60)
    print("                MÉTRICAS GLOBALES")
    print("="*60)
    
    # Precisión de decisión
    precision_global = pasos_correctos / total_pasos if total_pasos > 0 else 0
    print(f"Precisión de decisión global: {precision_global:.4f} ({precision_global*100:.2f}%)")
    print(f"  - Pasos correctos: {pasos_correctos}/{total_pasos}")
    
    # Cross Entropy
    ce_promedio = cross_entropy_total / total_pasos if total_pasos > 0 else 0
    print(f"Cross Entropy promedio: {ce_promedio:.4f}")
    
    # Alturas
    altura_hr_promedio = np.mean(alturas_hr) if alturas_hr else 0
    altura_modelo_promedio = np.mean(alturas_modelo) if alturas_modelo else 0
    print(f"Altura promedio HR: {altura_hr_promedio:.2f}")
    print(f"Altura promedio Modelo: {altura_modelo_promedio:.2f}")
    
    # Diferencias de altura
    if diferencias_altura:
        diferencia_promedio = np.mean(diferencias_altura)
        std_diferencia = np.std(diferencias_altura)
        mejora_promedio = -diferencia_promedio / altura_hr_promedio * 100 if altura_hr_promedio > 0 else 0
        print(f"Diferencia promedio de altura: {diferencia_promedio:.2f} ± {std_diferencia:.2f}")
        print(f"Mejora promedio vs HR: {mejora_promedio:.2f}%")
    
    # Comparación con random
    if alturas_random:
        altura_random_promedio = np.mean(alturas_random)
        print(f"Altura promedio Random: {altura_random_promedio:.2f}")
        if diferencias_altura_random:
            mejora_vs_random = -np.mean(diferencias_altura_random) / altura_random_promedio * 100
            print(f"Mejora modelo vs Random: {mejora_vs_random:.2f}%")
    
    # Exactitud de secuencia
    exactitud_secuencia = secuencias_exactas / total_secuencias if total_secuencias > 0 else 0
    print(f"Exactitud de secuencia: {secuencias_exactas}/{total_secuencias} ({exactitud_secuencia:.2%})")
    
    # Distribución de mejoras
    if diferencias_altura:
        mejores = sum(1 for diff in diferencias_altura if diff < 0)
        iguales = sum(1 for diff in diferencias_altura if diff == 0)
        peores = sum(1 for diff in diferencias_altura if diff > 0)
        
        print(f"\nDistribución de resultados:")
        print(f"  - Mejor que HR: {mejores}/{len(diferencias_altura)} ({mejores/len(diferencias_altura)*100:.1f}%)")
        print(f"  - Igual que HR: {iguales}/{len(diferencias_altura)} ({iguales/len(diferencias_altura)*100:.1f}%)")
        print(f"  - Peor que HR: {peores}/{len(diferencias_altura)} ({peores/len(diferencias_altura)*100:.1f}%)")
    
    print("="*60)

    hr.visualizar_packing(placements_hr, container_width=ancho, container_height=alto)
    hr.visualizar_packing(placements_modelo, container_width=ancho, container_height=alto)
    hr.visualizar_packing(placements_random, container_width=ancho, container_height=alto)  



    return {
        'precision_decision': precision_global,
        'cross_entropy': ce_promedio,
        'altura_hr_promedio': altura_hr_promedio,
        'altura_modelo_promedio': altura_modelo_promedio,
        'diferencia_altura_promedio': diferencia_promedio if diferencias_altura else 0,
        'exactitud_secuencia': exactitud_secuencia,
        'mejora_vs_hr': mejora_promedio if diferencias_altura else 0,
        'total_problemas': len(problemas)
    }


# Generar problemas de prueba para evaluación
problemas_evaluacion, ancho_eval, alto_eval = gen.generate_problems_guillotine(categoria, 10, export=False)
# problemas_evaluacion = gen.generate_problems_from_file("tests/c1p1.txt")
# ancho_eval, alto_eval = cat.CATEGORIES[categoria]["width"], cat.CATEGORIES[categoria]["height"]

print(f"\n=== EVALUACIÓN COMPLETA DEL MODELO ===")
metricas = evaluar_modelo_vs_hr(
    problemas=problemas_evaluacion,
    ancho=ancho_eval,
    alto=alto_eval,
    model=model,
    device="cpu",
    categoria=categoria
)

# # Guardar métricas en archivo
# import json
# with open(f'metricas_evaluacion_{categoria.lower()}.json', 'w') as f:
#     json.dump(metricas, f, indent=2)

# print(f"Métricas guardadas en metricas_evaluacion_{categoria.lower()}.json")