import random
import numpy as np
import hr_algorithm as hr


CATEGORIES = {
    "C1": {"num_items": 17, "width": 20, "height": 20},
    "C2": {"num_items": 25, "width": 40, "height": 15},
    "C3": {"num_items": 29, "width": 60, "height": 30},
    "C4": {"num_items": 49, "width": 60, "height": 60},
    "C5": {"num_items": 73, "width": 60, "height": 90},
    "C6": {"num_items": 97, "width": 80, "height": 120},
    "C7": {"num_items": 197, "width": 160, "height": 240},
}

def generate_problems_guillotine(category, n_problems=1, export=False):
    """
    Genera problemas usando el método de guillotina con rectángulos más extremos.
    """
    if category not in CATEGORIES:
        raise ValueError(f"Categoría no válida. Opciones: {list(CATEGORIES.keys())}")
    
    cat = CATEGORIES[category]
    problems = []
    
    def guillotine_cut_extreme(width, height, n_blocks, min_size=2):
        """
        Aplica cortes de guillotina para generar rectángulos más extremos y variados.
        """
        rectangles = [(0, 0, width, height)]
        
        for cut_num in range(n_blocks - 1):
            if not rectangles:
                break
                
            # Ordenar por área para preferir los más grandes
            rectangles.sort(key=lambda r: r[2] * r[3], reverse=True)
            
            cut_made = False
            for i, (x, y, w, h) in enumerate(rectangles):
                can_cut_vertical = w >= min_size * 2
                can_cut_horizontal = h >= min_size * 2
                
                if can_cut_vertical or can_cut_horizontal:
                    rectangles.pop(i)
                    
                    # Decidir tipo de corte con bias hacia cortes extremos
                    if can_cut_vertical and can_cut_horizontal:
                        # Favorecer el corte que cree rectangulos más alargados
                        aspect_ratio = w / h
                        if aspect_ratio > 2:  # Ya es ancho, hacer corte horizontal para crear más variedad
                            cut_vertical = random.choice([True, False])
                        elif aspect_ratio < 0.5:  # Ya es alto, hacer corte vertical
                            cut_vertical = random.choice([True, False])
                        else:
                            cut_vertical = random.choice([True, False])
                    elif can_cut_vertical:
                        cut_vertical = True
                    else:
                        cut_vertical = False
                    
                    if cut_vertical:
                        # Cortes más extremos para crear rectángulos alargados
                        # 30% de probabilidad de corte muy extremo
                        if random.random() < 0.3:
                            # Corte muy asimétrico (10-90% o 90-10%)
                            if random.choice([True, False]):
                                cut_pos = max(min_size, w // 10)  # Corte en 10%
                            else:
                                cut_pos = min(w - min_size, 9 * w // 10)  # Corte en 90%
                        else:
                            # Corte normal pero con más variabilidad
                            cut_min = max(min_size, w // 5)  # Más rango de corte
                            cut_max = min(w - min_size, 4 * w // 5)
                            if cut_min < cut_max:
                                cut_pos = random.randint(cut_min, cut_max)
                            else:
                                cut_pos = w // 2
                        
                        rect1 = (x, y, cut_pos, h)
                        rect2 = (x + cut_pos, y, w - cut_pos, h)
                    else:
                        # Cortes horizontales extremos
                        if random.random() < 0.3:
                            # Corte muy asimétrico
                            if random.choice([True, False]):
                                cut_pos = max(min_size, h // 10)
                            else:
                                cut_pos = min(h - min_size, 9 * h // 10)
                        else:
                            cut_min = max(min_size, h // 5)
                            cut_max = min(h - min_size, 4 * h // 5)
                            if cut_min < cut_max:
                                cut_pos = random.randint(cut_min, cut_max)
                            else:
                                cut_pos = h // 2
                        
                        rect1 = (x, y, w, cut_pos)
                        rect2 = (x, y + cut_pos, w, h - cut_pos)
                    
                    rectangles.extend([rect1, rect2])
                    cut_made = True
                    break
            
            if not cut_made:
                break
        
        return rectangles
    
    for i in range(n_problems):
        intentos = 0
        while intentos < 100:
            # Generar más rectángulos para tener más opciones
            target_rects = cat["num_items"]   # Más rectángulos para elegir
            
            rects_with_pos = guillotine_cut_extreme(
                cat["width"], cat["height"], 
                target_rects, 
                min_size=2
            )
            
            # print(f"Rectángulos generados: {len(rects_with_pos)}")
            # print(f"Área total: {sum(w*h for x,y,w,h in rects_with_pos)} / {cat['width']*cat['height']}")
            
            # Mostrar algunos ejemplos con sus aspect ratios
            for idx, (x, y, w, h) in enumerate(rects_with_pos[:10]):
                aspect = max(w/h, h/w)
            #     print(f"  Rect {idx+1}: pos=({x},{y}) size=({w},{h}) área={w*h} aspect={aspect:.2f}")
            # print("")

            # Filtros más permisivos para rectángulos extremos
            valid_rects = []
            for x, y, w, h in rects_with_pos:
                aspect = max(w/h, h/w) if min(w, h) > 0 else 1
                # Permitir aspect ratios hasta 7 y tamaños mínimos de 1
                if 1 <= min(w, h) and aspect <= 7:  # Más permisivo
                    valid_rects.append((w, h))
            
            # print(f"Rectángulos válidos después del filtro: {len(valid_rects)}")
            
            # Mostrar distribución de aspect ratios
            aspects = [max(w/h, h/w) for w, h in valid_rects]
            # print(f"Aspect ratios - Min: {min(aspects):.2f}, Max: {max(aspects):.2f}, Promedio: {np.mean(aspects):.2f}")
            
            if len(valid_rects) >= cat["num_items"]:
                # Contar frecuencias de cada rectángulo único
                from collections import Counter
                rect_counter = Counter(valid_rects)

                # Crear lista de rectángulos disponibles (máximo 2 de cada tipo)
                available_rects = []
                for rect, count in rect_counter.items():
                    # Agregar máximo 2 de cada tipo
                    available_rects.extend([rect] * min(count, 2))
                
                # print(f"Rectángulos únicos: {len(rect_counter)}")
                # print(f"Rectángulos disponibles (máx 2 por tipo): {len(available_rects)}")
                
                # Verificar si tenemos suficientes rectángulos únicos
                if len(available_rects) >= cat["num_items"]:
                    # Seleccionar con bias hacia rectángulos más extremos
                    available_rects_sorted = sorted(available_rects, key=lambda r: max(r[0]/r[1], r[1]/r[0]), reverse=True)
                    
                    # Tomar 50% de los más extremos, 50% aleatorio
                    n_extreme = cat["num_items"] // 2
                    n_random = cat["num_items"] - n_extreme
                    
                    extreme_rects = available_rects_sorted[:n_extreme] if n_extreme > 0 else []
                    remaining_rects = available_rects_sorted[n_extreme:]
                    
                    if len(remaining_rects) >= n_random:
                        random_rects = random.sample(remaining_rects, n_random)
                    else:
                        random_rects = remaining_rects
                    
                    final_rects = extreme_rects + random_rects
                    
                    # Si no tenemos suficientes, completar con los disponibles
                    if len(final_rects) < cat["num_items"]:
                        remaining_needed = cat["num_items"] - len(final_rects)
                        used_rects = set(final_rects)
                        additional_candidates = [r for r in available_rects if r not in used_rects or final_rects.count(r) < 3]
                        additional = random.sample(additional_candidates, min(remaining_needed, len(additional_candidates)))
                        final_rects.extend(additional)
                    
                    final_rects = final_rects[:cat["num_items"]]  # Asegurar número exacto
                    
                    # Verificar distribución final
                    final_counter = Counter(final_rects)
                    max_repetitions = max(final_counter.values())
                    # print(f"Máximo de repeticiones en resultado final: {max_repetitions}")
                    
                    if max_repetitions <= 3:
                        problems.append(final_rects)
                        
                        # Mostrar estadísticas de los rectángulos seleccionados
                        final_aspects = [max(w/h, h/w) for w, h in final_rects]
                        final_sizes = [min(w, h) for w, h in final_rects]
                        # print(f"Rectángulos finales - Aspect ratios: Min={min(final_aspects):.2f}, Max={max(final_aspects):.2f}")
                        # print(f"Tamaños mínimos: Min={min(final_sizes)}, Max={max(final_sizes)}")
                        # print(f"Distribución de rectángulos: {dict(final_counter)}")
                        
                        if export:
                            fname = f"{category.lower()}_extreme_{i+1}.txt"
                            with open(fname, "w") as f:
                                for w, h in final_rects:
                                    f.write(f"{h}\t{w}\n")
                            
                            total_area_selected = sum(w*h for w,h in final_rects)
                            total_area_generated = sum(w*h for x,y,w,h in rects_with_pos)
                            container_area = cat['width'] * cat['height']
                            
                            # print(f"Exportado: {fname}")
                            # print(f"  Área contenedor: {container_area}")
                            # print(f"  Área generada total: {total_area_generated}")
                            # print(f"  Área seleccionada: {total_area_selected}")
                            # print(f"  Cobertura: {total_area_generated/container_area:.1%}")
                        break
                    # else:
                        # print(f"Demasiadas repeticiones ({max_repetitions}), reintentando...")
                # else:
                    # print(f"No hay suficientes rectángulos únicos. Disponibles: {len(available_rects)}, Necesarios: {cat['num_items']}")


            intentos += 1
        
        if intentos >= 100:
            print(f"Advertencia: No se pudo generar el problema {i+1} tras 100 intentos")
    
    # Visualizar solo si se generaron problemas
    if problems and rects_with_pos:
        hr.visualizar_packing(
            [(rect[2:], (rect[0], rect[1])) for rect in rects_with_pos],
            container_width=cat["width"], 
            container_height=cat["height"], 
            show=True
        )
    
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
