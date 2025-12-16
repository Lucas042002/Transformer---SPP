"""Algoritmos greedy clásicos para Strip Packing Problem.

Este módulo implementa algoritmos baseline para comparación con el modelo Pointer:
- FFDH (First Fit Decreasing Height)
- BFDH (Best Fit Decreasing Height)
- Next Fit (NF)
- Bottom-Left (BL)

Todos los algoritmos retornan:
    placements: List[(x, y, w, h, rotated)] - Coordenadas de cada rectángulo colocado
    altura: float - Altura total del packing
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

from typing import List, Tuple
import time


# ============================================================================
# FIRST FIT DECREASING HEIGHT (FFDH)
# ============================================================================
def ffdh_packing(rects: List[Tuple[int, int]], container_width: int) -> Tuple[List[Tuple], float]:
    """First Fit Decreasing Height - Baseline clásico en literatura.
    
    Algoritmo:
        1. Ordena rectángulos por altura descendente
        2. Para cada rectángulo, busca el PRIMER nivel donde quepa
        3. Si no cabe en ningún nivel, crea uno nuevo
    
    Args:
        rects: Lista de rectángulos [(w, h), ...]
        container_width: Ancho del contenedor
        
    Returns:
        placements: Lista de (x, y, w, h, rotated) para cada rectángulo
        altura: Altura total del packing
    """
    if not rects:
        return [], 0.0
    
    # Ordenar por altura descendente (más alto primero)
    sorted_rects = sorted(enumerate(rects), key=lambda x: x[1][1], reverse=True)
    
    # Niveles: [(y_base, altura_nivel, ancho_ocupado)]
    levels = []
    placements = [None] * len(rects)  # Mantener orden original
    
    for orig_idx, (w, h) in sorted_rects:
        placed = False
        
        # Buscar PRIMER nivel donde quepa
        for level_idx, (y_base, level_h, x_used) in enumerate(levels):
            if h <= level_h and w <= (container_width - x_used):
                # Cabe en este nivel
                placements[orig_idx] = (x_used, y_base, w, h, False)
                levels[level_idx] = (y_base, level_h, x_used + w)
                placed = True
                break
        
        # Si no cabe, crear nivel nuevo
        if not placed:
            y_new = sum(level[1] for level in levels)  # Suma de alturas de niveles previos
            placements[orig_idx] = (0, y_new, w, h, False)
            levels.append((y_new, h, w))
    
    altura_final = sum(level[1] for level in levels)
    return placements, altura_final


# ============================================================================
# BEST FIT DECREASING HEIGHT (BFDH)
# ============================================================================
def bfdh_packing(rects: List[Tuple[int, int]], container_width: int) -> Tuple[List[Tuple], float]:
    """Best Fit Decreasing Height - Variante mejorada de FFDH.
    
    Algoritmo:
        1. Ordena rectángulos por altura descendente
        2. Para cada rectángulo, busca el nivel con MENOR espacio residual donde quepa
        3. Minimiza el desperdicio por nivel
    
    Args:
        rects: Lista de rectángulos [(w, h), ...]
        container_width: Ancho del contenedor
        
    Returns:
        placements: Lista de (x, y, w, h, rotated) para cada rectángulo
        altura: Altura total del packing
    """
    if not rects:
        return [], 0.0
    
    # Ordenar por altura descendente
    sorted_rects = sorted(enumerate(rects), key=lambda x: x[1][1], reverse=True)
    
    # Niveles: [(y_base, altura_nivel, ancho_ocupado)]
    levels = []
    placements = [None] * len(rects)
    
    for orig_idx, (w, h) in sorted_rects:
        # Buscar el MEJOR nivel (menor espacio residual)
        best_level = -1
        best_waste = float('inf')
        
        for level_idx, (y_base, level_h, x_used) in enumerate(levels):
            if h <= level_h and w <= (container_width - x_used):
                # Espacio residual en este nivel
                waste = (container_width - x_used) - w
                if waste < best_waste:
                    best_waste = waste
                    best_level = level_idx
        
        # Colocar en el mejor nivel encontrado
        if best_level >= 0:
            y_base, level_h, x_used = levels[best_level]
            placements[orig_idx] = (x_used, y_base, w, h, False)
            levels[best_level] = (y_base, level_h, x_used + w)
        else:
            # Crear nivel nuevo
            y_new = sum(level[1] for level in levels)
            placements[orig_idx] = (0, y_new, w, h, False)
            levels.append((y_new, h, w))
    
    altura_final = sum(level[1] for level in levels)
    return placements, altura_final


# ============================================================================
# NEXT FIT (NF)
# ============================================================================
def next_fit_packing(rects: List[Tuple[int, int]], container_width: int) -> Tuple[List[Tuple], float]:
    """Next Fit - Algoritmo greedy más simple (baseline trivial).
    
    Algoritmo:
        1. Mantiene solo 1 nivel activo
        2. Intenta colocar rectángulo en nivel actual
        3. Si no cabe, cierra nivel y crea uno nuevo
    
    Args:
        rects: Lista de rectángulos [(w, h), ...]
        container_width: Ancho del contenedor
        
    Returns:
        placements: Lista de (x, y, w, h, rotated) para cada rectángulo
        altura: Altura total del packing
    """
    if not rects:
        return [], 0.0
    
    placements = []
    current_y = 0
    current_x = 0
    current_level_height = 0
    
    for w, h in rects:
        # Si no cabe en el nivel actual, crear nuevo nivel
        if current_x + w > container_width:
            current_y += current_level_height
            current_x = 0
            current_level_height = 0
        
        # Colocar rectángulo
        placements.append((current_x, current_y, w, h, False))
        current_x += w
        current_level_height = max(current_level_height, h)
    
    altura_final = current_y + current_level_height
    return placements, altura_final


# ============================================================================
# BOTTOM-LEFT (BL)
# ============================================================================
def bottom_left_packing(rects: List[Tuple[int, int]], container_width: int) -> Tuple[List[Tuple], float]:
    """Bottom-Left - Algoritmo clásico en literatura académica.
    
    Algoritmo:
        1. Para cada rectángulo (ordenado por área descendente)
        2. Encuentra la posición más baja posible
        3. Si hay empate, elige la más a la izquierda
        4. Verifica que no haya solapamiento
    
    Args:
        rects: Lista de rectángulos [(w, h), ...]
        container_width: Ancho del contenedor
        
    Returns:
        placements: Lista de (x, y, w, h, rotated) para cada rectángulo
        altura: Altura total del packing
    """
    if not rects:
        return [], 0.0
    
    # Ordenar por área descendente (heurística común)
    sorted_rects = sorted(enumerate(rects), key=lambda x: x[1][0] * x[1][1], reverse=True)
    

    placements = [None] * len(rects)
    placed_rects = []  # Lista de (x, y, w, h) ya colocados
    
    for orig_idx, (w, h) in sorted_rects:
        # Buscar la posición más baja-izquierda donde quepa
        best_pos = None
        best_y = float('inf')
        best_x = float('inf')
        
        # Probar posiciones candidatas (esquinas de rectángulos existentes + origen)
        candidates = [(0, 0)]
        for px, py, pw, ph in placed_rects:
            candidates.append((px + pw, py))  # Derecha
            candidates.append((px, py + ph))  # Arriba
        
        for x, y in candidates:
            # Verificar que quepa en el contenedor
            if x + w > container_width:
                continue
            
            # Verificar que no se solape con rectángulos existentes
            overlaps = False
            for px, py, pw, ph in placed_rects:
                if not (x + w <= px or x >= px + pw or y + h <= py or y >= py + ph):
                    overlaps = True
                    break
            
            if not overlaps:
                # Bajar el rectángulo lo máximo posible
                max_y_below = 0
                for px, py, pw, ph in placed_rects:
                    # Si hay solapamiento horizontal
                    if not (x + w <= px or x >= px + pw):
                        # Y el rectángulo está debajo de la posición actual
                        if py + ph <= y:
                            max_y_below = max(max_y_below, py + ph)
                
                actual_y = max(y, max_y_below)
                
                # Verificar nuevamente si hay solapamiento en la nueva posición
                overlaps_final = False
                for px, py, pw, ph in placed_rects:
                    if not (x + w <= px or x >= px + pw or actual_y + h <= py or actual_y >= py + ph):
                        overlaps_final = True
                        break
                
                if not overlaps_final:
                    # Esta es una posición válida, ver si es mejor
                    if actual_y < best_y or (actual_y == best_y and x < best_x):
                        best_y = actual_y
                        best_x = x
                        best_pos = (x, actual_y)
        
        # Colocar en la mejor posición encontrada
        if best_pos:
            x, y = best_pos
            placements[orig_idx] = (x, y, w, h, False)
            placed_rects.append((x, y, w, h))
        else:
            # Fallback: colocar arriba de todo
            max_height = max((py + ph for _, py, _, ph in placed_rects), default=0)
            placements[orig_idx] = (0, max_height, w, h, False)
            placed_rects.append((0, max_height, w, h))
    
    # Calcular altura final
    if placed_rects:
        altura_final = max(py + ph for _, py, _, ph in placed_rects)
    else:
        altura_final = 0.0
    
    return placements, altura_final


# ============================================================================
# FUNCIÓN DE COMPARACIÓN
# ============================================================================
def comparar_algoritmos(rects: List[Tuple[int, int]], container_width: int, 
                        algoritmos: List[str] = None) -> dict:
    """Ejecuta y compara múltiples algoritmos greedy.
    
    Args:
        rects: Lista de rectángulos [(w, h), ...]
        container_width: Ancho del contenedor
        algoritmos: Lista de nombres ['ffdh', 'bfdh', 'nf', 'bl'] o None para todos
        
    Returns:
        Dict con resultados: {
            'ffdh': {'placements': [...], 'altura': float, 'tiempo_ms': float},
            'bfdh': {...},
            ...
        }
    """
    if algoritmos is None:
        algoritmos = ['nf', 'ffdh', 'bfdh', 'bl']
    
    algoritmos_map = {
        'nf': ('Next Fit', next_fit_packing),
        'ffdh': ('First Fit Decreasing Height', ffdh_packing),
        'bfdh': ('Best Fit Decreasing Height', bfdh_packing),
        'bl': ('Bottom-Left', bottom_left_packing),
    }
    
    resultados = {}
    
    for alg_key in algoritmos:
        if alg_key not in algoritmos_map:
            continue
        
        nombre, func = algoritmos_map[alg_key]
        
        start = time.time()
        placements, altura = func(rects, container_width)
        tiempo_ms = (time.time() - start) * 1000
        
        resultados[alg_key] = {
            'nombre': nombre,
            'placements': placements,
            'altura': altura,
            'tiempo_ms': tiempo_ms,
            'n_colocados': len([p for p in placements if p is not None])
        }
    
    return resultados


# ============================================================================
# DEMO / TESTING
# ============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path
    import generator as gen
    
    # Configuración
    CATEGORIA = "C5"  # Puedes cambiar a C2, C3, etc.
    
    print("="*80)
    print(f"COMPARACIÓN DE ALGORITMOS GREEDY - Problema Generado {CATEGORIA}")
    print("="*80)
    
    # Generar problema usando el generador
    problems = gen.generate_problems_guillotine(CATEGORIA, n_problems=1)
    rects_test = problems[0][0]
    print(f"Número de rectángulos: {len(rects_test)}")
    W = gen.CATEGORIES[CATEGORIA]["width"]
    
    print(f"\nCategoría: {CATEGORIA}")
    print(f"Problema: {len(rects_test)} rectángulos, W={W}")
    print(f"Área total: {sum(w*h for w, h in rects_test)}")
    print(f"Cota inferior teórica: {sum(w*h for w, h in rects_test) / W:.1f}")
    
    # Ejecutar algoritmos
    resultados = comparar_algoritmos(rects_test, W)
    
    print(f"\n{'Algoritmo':<35} {'Altura':<10} {'Tiempo':<15} {'vs Mejor':<10}")
    print("-"*80)
    
    mejor_altura = min(res['altura'] for res in resultados.values())
    
    for alg_key in ['nf', 'ffdh', 'bfdh', 'bl']:
        res = resultados[alg_key]
        diff_pct = ((res['altura'] - mejor_altura) / mejor_altura * 100) if mejor_altura > 0 else 0
        print(f"{res['nombre']:<35} {res['altura']:<10.1f} {res['tiempo_ms']:<15.3f}ms {diff_pct:>9.1f}%")
    
    # Visualizar los 4 algoritmos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Comparación Algoritmos Greedy - Categoría {CATEGORIA} (W={W})', fontsize=16, fontweight='bold')
    
    algoritmos_orden = ['ffdh', 'bfdh', 'nf', 'bl']
    colores = plt.cm.Set3(range(len(rects_test)))
    
    for idx, alg_key in enumerate(algoritmos_orden):
        ax = axes[idx // 2, idx % 2]
        res = resultados[alg_key]
        
        # Dibujar contenedor
        ax.add_patch(patches.Rectangle((0, 0), W, res['altura'], 
                                      fill=False, edgecolor='black', linewidth=2))
        
        # Dibujar rectángulos
        for i, placement in enumerate(res['placements']):
            if placement:
                x, y, w, h, _ = placement
                rect = patches.Rectangle((x, y), w, h, 
                                        facecolor=colores[i % len(colores)],
                                        edgecolor='black', linewidth=0.5, alpha=0.7)
                ax.add_patch(rect)
                
                # Etiqueta con índice
                ax.text(x + w/2, y + h/2, str(i+1), 
                       ha='center', va='center', fontsize=6, fontweight='bold')
        
        # Configurar ejes
        ax.set_xlim(-1, W + 1)
        ax.set_ylim(-1, res['altura'] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Ancho', fontsize=10)
        ax.set_ylabel('Altura', fontsize=10)
        
        # Título con estadísticas
        diff_pct = ((res['altura'] - mejor_altura) / mejor_altura * 100) if mejor_altura > 0 else 0
        ax.set_title(f"{res['nombre']}\nAltura: {res['altura']:.1f} | Tiempo: {res['tiempo_ms']:.2f}ms | vs Mejor: +{diff_pct:.1f}%",
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar imagen
    output_file = Path(__file__).parent / "img" / f"comparacion_greedy_{CATEGORIA}.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualizacion guardada en: {output_file}")
    
    plt.show()
