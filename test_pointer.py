"""
Script de testing para el modelo pointer.

Verifica:
1. Carga correcta del modelo entrenado
2. Flujo de datos correcto (shapes, tipos, valores)
3. Inferencia con hr_pointer
4. Comparación de resultados vs HR puro
5. Métricas de desempeño
6. Visualización de resultados
"""
import torch
import os
from typing import List, Tuple, Dict, Optional
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

import generator as gen
import categories as cat
import hr_algorithm as hr
import hr_pointer as hrp
from pointer_model import SPPPointerModel
import greedy_algorithms as greedy


def _dibujar_packing_en_axes(ax, placements, container_width, container_height, titulo="Packing"):
    """Dibuja el packing en un axes específico (para comparaciones lado a lado).
    
    Args:
        ax: Axes de matplotlib donde dibujar
        placements: Lista de (rect, position)
        container_width: Ancho del contenedor
        container_height: Altura del contenedor
        titulo: Título del subplot
    """
    colors = {}
    
    used_heights = [pos[1] + rect[1] for rect, pos in placements]
    max_height = max(used_heights) if used_heights else 0
    
    # Configurar axes
    ax.set_xlim(0, container_width)
    ax.set_ylim(0, max_height)
    ax.set_aspect('equal')
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel("Ancho")
    ax.set_ylabel("Altura")
    
    # Ejes con números enteros
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Dibujar rectángulos
    for i, (rect, pos) in enumerate(placements):
        w, h = rect
        x, y = pos
        color = colors.get(rect)
        if color is None:
            color = [random.random() for _ in range(3)]
            colors[rect] = color
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor=color))
        ax.text(x + w/2, y + h/2, f"{i}", fontsize=8, ha='center', va='center', color='white', fontweight='bold')
    
    # Línea roja en la altura del contenedor
    if container_height:
        ax.axhline(y=container_height, color='red', linestyle='--', linewidth=2, label='Altura contenedor')
    
    ax.grid(True, alpha=0.3)


def cargar_modelo_entrenado(model_path: str, device: str = "cpu") -> SPPPointerModel:
    """Carga un modelo pointer entrenado desde un checkpoint.
    
    Args:
        model_path: Ruta al archivo .pth del modelo
        device: Dispositivo ('cpu' o 'cuda')
        
    Returns:
        Modelo cargado y en modo evaluación
    """
    print(f"\n{'='*80}")
    print(f"CARGANDO MODELO")
    print(f"{'='*80}")
    print(f"Archivo: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraer hiperparámetros del checkpoint
    categoria = checkpoint.get('categoria', 'C1')
    num_enc_layers = checkpoint.get('num_enc_layers', 4)
    num_heads = checkpoint.get('num_heads', 8)
    max_val_acc = checkpoint.get('max_val_acc', 0.0)
    
    print(f"Categoría: {categoria}")
    print(f"Encoder layers: {num_enc_layers}")
    print(f"Attention heads: {num_heads}")
    print(f"Accuracy de validación: {max_val_acc:.4f}")
    
    # Crear modelo con la misma arquitectura
    model = SPPPointerModel(
        d_model=256,
        rect_feat_dim=10,
        space_feat_dim=19,  # Actualizado: 10 básicas + 1 fragmentation + 8 compatibilidad (incluye minimización altura)
        num_enc_layers=num_enc_layers,
        num_heads=num_heads,
        d_ff=512,
        dropout=0.1,
        max_steps=cat.CATEGORIES[categoria]["num_items"]
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total_params:,}")
    print(f"Modelo cargado exitosamente")
    
    return model, categoria


def validar_flujo_datos(
    model: SPPPointerModel,
    problema_rects: List[Tuple[int, int]],
    categoria: str,
    device: str = "cpu"
):
    """Valida que el flujo de datos sea correcto durante la inferencia.
    
    Verifica shapes, tipos y valores en cada paso del procesamiento.
    """
    print(f"\n{'='*80}")
    print(f"VALIDACIÓN DE FLUJO DE DATOS")
    print(f"{'='*80}")
    
    W = cat.CATEGORIES[categoria]["width"]
    Href = cat.CATEGORIES[categoria]["height"]
    
    print(f"Problema: {len(problema_rects)} rectángulos")
    print(f"Contenedor: {W}x{Href}")
    
    # ============================================================
    # PASO 1: Inicializar cache del encoder
    # ============================================================
    print(f"\n--- PASO 1: Inicializar Cache Encoder ---")
    
    rects_originales = problema_rects.copy()
    spaces = [(0, 0, W, 1000)]
    
    cached_rect_enc, _ = hrp.inicializar_cache_encoder(
        rects_originales, spaces, W, categoria, model, device
    )
    
    print(f"Cached rect_enc shape: {cached_rect_enc.shape}")
    assert cached_rect_enc.shape == (1, len(problema_rects), 256), \
        f"Shape incorrecto! Esperado (1, {len(problema_rects)}, 256), got {cached_rect_enc.shape}"
    
    # Verificar que no hay NaN o Inf
    assert not torch.isnan(cached_rect_enc).any(), "Encoder contiene NaN!"
    assert not torch.isinf(cached_rect_enc).any(), "Encoder contiene Inf!"
    print(f"Sin NaN/Inf en embeddings")
    
    # ============================================================
    # PASO 2: Simular una decisión (primer paso)
    # ============================================================
    print(f"\n--- PASO 2: Simular Decisión (Paso 0) ---")
    
    remaining_rects = problema_rects.copy()
    active_space = spaces[0]
    step = 0
    
    rect_idx, rect_elegido = hrp.usar_modelo_pointer_para_decision_optimized(
        remaining_rects=remaining_rects,
        active_space=active_space,
        model=model,
        container_width=W,
        category=categoria,
        device=device,
        step=step,
        rects_originales=rects_originales,
        cached_rect_enc=cached_rect_enc,
        cached_global_ctx=None
    )
    
    print(f"Rectángulos disponibles: {len(remaining_rects)}")
    print(f"Espacio activo: {active_space}")
    print(f"Rectángulo elegido: índice={rect_idx}, rect={rect_elegido}")
    
    assert rect_idx >= 0, "No se eligió ningún rectángulo!"
    assert rect_elegido is not None, "Rectángulo elegido es None!"
    assert rect_elegido in remaining_rects, "Rectángulo elegido no está en remaining_rects!"
    print(f"Decisión válida")
    
    # ============================================================
    # PASO 3: Verificar features intermedias
    # ============================================================
    print(f"\n--- PASO 3: Verificar Features Intermedias ---")
    
    import states as st
    
    # Features de rectángulos (10 dims)
    rect_feats_muestra = st._rect_features_optimized(
        rect_elegido, W, Href, rects_originales
    )
    print(f"Rect features (muestra): {len(rect_feats_muestra)} dims")
    assert len(rect_feats_muestra) == 10, f"Esperado 10 dims, got {len(rect_feats_muestra)}"
    print(f"  Valores: min={min(rect_feats_muestra):.3f}, max={max(rect_feats_muestra):.3f}")
    print(f"Rect features correcto (10 dims)")
    
    # Features de espacio (19 dims) - Actualizado con features de minimización de altura
    current_max_height = active_space[1] + active_space[3]
    space_feat_muestra = st._space_features_optimized(
        active_space, W, Href, spaces, current_max_height, remaining_rects=remaining_rects
    )
    print(f"Space features (muestra): {len(space_feat_muestra)} dims")
    assert len(space_feat_muestra) == 19, f"Esperado 19 dims, got {len(space_feat_muestra)}"
    print(f"  Valores: min={min(space_feat_muestra):.3f}, max={max(space_feat_muestra):.3f}")
    print(f"Space features correcto (19 dims)")
    
    # ============================================================
    # PASO 4: Verificar máscaras de factibilidad
    # ============================================================
    print(f"\n--- PASO 4: Verificar Máscaras de Factibilidad ---")
    
    fits_count = 0
    for r in remaining_rects:
        rw, rh = r
        _, _, sw, sh = active_space
        fits_normal = (rw <= sw and rh <= sh)
        fits_rotated = (rh <= sw and rw <= sh)
        if fits_normal or fits_rotated:
            fits_count += 1
    
    print(f"Rectángulos que caben: {fits_count}/{len(remaining_rects)}")
    assert fits_count > 0, "Ningún rectángulo cabe en el espacio!"
    print(f"Máscara de factibilidad válida")
    
    print(f"\n{'='*80}")
    print(f"VALIDACIÓN COMPLETA - TODO CORRECTO")
    print(f"{'='*80}")


def test_problema_completo(
    model: SPPPointerModel,
    problema_rects: List[Tuple[int, int]],
    categoria: str,
    device: str = "cpu",
    verbose: bool = True
) -> Dict:
    """Ejecuta inferencia completa en un problema y devuelve resultados.
    
    Returns:
        Dict con:
            - placements: lista de (rect, pos)
            - altura: altura final alcanzada
            - rect_sequence: secuencia de rectángulos colocados
            - tiempo_ms: tiempo de ejecución en milisegundos
            - exito: si se colocaron todos los rectángulos
    """
    W = cat.CATEGORIES[categoria]["width"]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"TEST DE PROBLEMA COMPLETO")
        print(f"{'='*80}")
        print(f"Rectángulos: {len(problema_rects)}")
        print(f"Contenedor: {W}x∞")
    
    # Ejecutar inferencia con medición de tiempo
    start_time = time.time()
    
    placements, altura, rect_sequence, _ = hrp.heuristic_recursion_pointer(
        rects=problema_rects,
        container_width=W,
        model=model,
        category=categoria,
        device=device
    )
    
    tiempo_ms = (time.time() - start_time) * 1000
    
    exito = len(placements) == len(problema_rects)
    
    if verbose:
        print(f"\nRESULTADOS:")
        print(f"  Rectángulos colocados: {len(placements)}/{len(problema_rects)}")
        print(f"  Altura alcanzada: {altura}")
        print(f"  Tiempo de ejecución: {tiempo_ms:.2f} ms")
        print(f"  Estado: {'ÉXITO' if exito else 'FALLÓ'}")
        
        if not exito:
            print(f"\nADVERTENCIA: No se colocaron todos los rectángulos")
            print(f"  Faltaron: {len(problema_rects) - len(placements)}")
    
    return {
        'placements': placements,
        'altura': altura,
        'rect_sequence': rect_sequence,
        'tiempo_ms': tiempo_ms,
        'exito': exito,
        'n_rects': len(problema_rects),
        'n_colocados': len(placements)
    }


def visualizar_resultado_pointer(
    placements: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    categoria: str,
    titulo: str = "Resultado Modelo Pointer",
    show: bool = True
):
    """Visualiza el resultado del packing.
    
    Args:
        placements: Lista de (rect, position) con los rectángulos colocados
        categoria: Categoría del problema (para obtener W y H)
        titulo: Título del gráfico
        show: Si True, muestra el gráfico inmediatamente
        
    Returns:
        Figura de matplotlib
    """
    W = cat.CATEGORIES[categoria]["width"]
    H = cat.CATEGORIES[categoria]["height"]
    
    # Crear figura y dibujar packing
    fig, ax = plt.subplots(figsize=(10, 10))
    _dibujar_packing_en_axes(ax, placements, W, H, titulo)
    
    plt.tight_layout()
    if show:
        plt.show()
    
    return fig


def comparar_con_hr_puro(
    problema_rects: List[Tuple[int, int]],
    model: SPPPointerModel,
    categoria: str,
    device: str = "cpu",
    visualizar: bool = False
):
    """Compara resultados del modelo pointer vs HR puro.
    
    Muestra diferencias en altura, tiempo de ejecución, etc.
    Opcionalmente muestra visualización lado a lado.
    
    Args:
        problema_rects: Lista de rectángulos a empaquetar
        model: Modelo pointer entrenado
        categoria: Categoría del problema
        device: 'cpu' o 'cuda'
        visualizar: Si True, muestra visualización comparativa
    """
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN: POINTER MODEL vs HR PURO")
    print(f"{'='*80}")
    
    W = cat.CATEGORIES[categoria]["width"]
    
    # ============ HR PURO ============
    print(f"\nEjecutando HR Puro...")
    start_hr = time.time()
    placements_hr, altura_hr, rect_seq_hr, _ = hr.heuristic_recursion(
        rects=problema_rects.copy(),
        container_width=W,
        category=categoria
    )
    tiempo_hr_ms = (time.time() - start_hr) * 1000
    
    # ============ POINTER MODEL ============
    print(f"Ejecutando Pointer Model...")
    start_pointer = time.time()
    placements_pointer, altura_pointer, rect_seq_pointer, _ = hrp.heuristic_recursion_pointer(
        rects=problema_rects.copy(),
        container_width=W,
        model=model,
        category=categoria,
        device=device
    )
    tiempo_pointer_ms = (time.time() - start_pointer) * 1000
    
    # ============ COMPARACIÓN ============
    print(f"\nRESULTADOS COMPARATIVOS:")
    print(f"\n{'Métrica':<25} {'HR Puro':<15} {'Pointer':<15} {'Diferencia':<15}")
    print(f"{'-'*70}")
    
    # Altura
    diff_altura = altura_pointer - altura_hr
    perc_altura = (diff_altura / altura_hr * 100) if altura_hr > 0 else 0
    print(f"{'Altura':<25} {altura_hr:<15} {altura_pointer:<15} {diff_altura:+.0f} ({perc_altura:+.1f}%)")
    
    # Tiempo
    diff_tiempo = tiempo_pointer_ms - tiempo_hr_ms
    perc_tiempo = (diff_tiempo / tiempo_hr_ms * 100) if tiempo_hr_ms > 0 else 0
    print(f"{'Tiempo (ms)':<25} {tiempo_hr_ms:<15.2f} {tiempo_pointer_ms:<15.2f} {diff_tiempo:+.2f} ({perc_tiempo:+.1f}%)")
    
    # Rectángulos colocados
    n_hr = len(placements_hr)
    n_pointer = len(placements_pointer)
    diff_rects = n_pointer - n_hr
    print(f"{'Rectángulos colocados':<25} {n_hr:<15} {n_pointer:<15} {diff_rects:+}")
    
    # ============ ANÁLISIS ============
    print(f"\nANÁLISIS:")
    
    if altura_pointer < altura_hr:
        print(f"  Pointer es MEJOR en altura ({abs(perc_altura):.1f}% más compacto)")
    elif altura_pointer == altura_hr:
        print(f"  Pointer IGUALA al HR en altura")
    else:
        print(f"  Pointer es PEOR en altura ({perc_altura:.1f}% más alto)")
    
    if tiempo_pointer_ms < tiempo_hr_ms:
        print(f"  Pointer es MÁS RÁPIDO ({abs(perc_tiempo):.1f}% menos tiempo)")
    else:
        print(f"  Pointer es más lento ({perc_tiempo:.1f}% más tiempo)")
    
    if n_pointer == n_hr == len(problema_rects):
        print(f"  Ambos colocaron todos los rectángulos")
    elif n_pointer < n_hr:
        print(f"  Pointer colocó MENOS rectángulos que HR")
    
    # ============ VISUALIZACIÓN ============
    if visualizar:
        print(f"\nGenerando visualización comparativa...")
        
        # Crear figura con 2 subplots lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        H = cat.CATEGORIES[categoria]["height"]
        
        # HR Puro (izquierda)
        _dibujar_packing_en_axes(
            ax=ax1,
            placements=placements_hr,
            container_width=W,
            container_height=H,
            titulo=f"HR Puro\nAltura: {altura_hr} | Tiempo: {tiempo_hr_ms:.1f}ms"
        )
        
        # Pointer Model (derecha)
        _dibujar_packing_en_axes(
            ax=ax2,
            placements=placements_pointer,
            container_width=W,
            container_height=H,
            titulo=f"Pointer Model\nAltura: {altura_pointer} | Tiempo: {tiempo_pointer_ms:.1f}ms"
        )
        
        # Título general con diferencia
        diff_str = f"{diff_altura:+}" if diff_altura != 0 else "IGUAL"
        fig.suptitle(
            f"Comparación: {len(problema_rects)} rectángulos | Diferencia altura: {diff_str}",
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
    
    return {
        'hr': {'altura': altura_hr, 'tiempo_ms': tiempo_hr_ms, 'n_colocados': n_hr, 'placements': placements_hr},
        'pointer': {'altura': altura_pointer, 'tiempo_ms': tiempo_pointer_ms, 'n_colocados': n_pointer, 'placements': placements_pointer}
    }


def test_suite_completo(
    model_path: str,
    n_problemas_test: int = 10,
    categoria: str = "C1",
    device: str = "cpu"
):
    """Suite de testing completo para el modelo pointer.
    
    Args:
        model_path: Ruta al modelo .pth
        n_problemas_test: Número de problemas aleatorios para probar
        categoria: Categoría de problemas
        device: 'cpu' o 'cuda'
    """
    print(f"\n{'='*80}")
    print(f"TEST SUITE COMPLETO - MODELO POINTER")
    print(f"{'='*80}")
    
    # ============================================================
    # 1. CARGAR MODELO
    # ============================================================
    model, cat_checkpoint = cargar_modelo_entrenado(model_path, device)
    
    # Verificar que la categoría coincida
    if cat_checkpoint != categoria:
        print(f"\nADVERTENCIA: Modelo entrenado en {cat_checkpoint}, testeando en {categoria}")
    
    # ============================================================
    # 2. GENERAR PROBLEMAS DE TEST
    # ============================================================
    print(f"\n{'='*80}")
    print(f"GENERANDO PROBLEMAS DE TEST")
    print(f"{'='*80}")
    print(f"Cantidad: {n_problemas_test}")
    print(f"Categoría: {categoria}")
    
    problems, W, H = gen.generate_problems_guillotine(categoria, n_problemas_test, export=False)
    print(f"{len(problems)} problemas generados")
    
    # ============================================================
    # 3. VALIDAR FLUJO DE DATOS (primer problema)
    # ============================================================
    validar_flujo_datos(model, problems[0], categoria, device)
    
    # ============================================================
    # 4. TEST EN TODOS LOS PROBLEMAS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"TESTING EN {n_problemas_test} PROBLEMAS")
    print(f"{'='*80}")
    
    resultados = []
    exitos = 0
    tiempos = []
    alturas_hr = []
    alturas_pointer = []
    
    for idx, rects in enumerate(problems):
        print(f"\n--- Problema {idx+1}/{n_problemas_test} ---")
        
        resultado = test_problema_completo(
            model, rects, categoria, device, verbose=False
        )
        resultados.append(resultado)
        
        if resultado['exito']:
            exitos += 1
        
        tiempos.append(resultado['tiempo_ms'])
        
        # Comparar con HR
        _, altura_hr, _, _ = hr.heuristic_recursion(rects, W, categoria)
        alturas_hr.append(altura_hr)
        alturas_pointer.append(resultado['altura'])
        
        # Mostrar resultado compacto
        status = "OK" if resultado['exito'] else "FALLÓ"
        diff_altura = resultado['altura'] - altura_hr
        print(f"  {status} Colocados: {resultado['n_colocados']}/{resultado['n_rects']} | "
              f"Altura: {resultado['altura']} (HR: {altura_hr}, diff: {diff_altura:+}) | "
              f"Tiempo: {resultado['tiempo_ms']:.1f}ms")
    
    # ============================================================
    # 5. ESTADÍSTICAS FINALES
    # ============================================================
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS FINALES")
    print(f"{'='*80}")
    
    tasa_exito = (exitos / n_problemas_test) * 100
    tiempo_promedio = sum(tiempos) / len(tiempos)
    tiempo_min = min(tiempos)
    tiempo_max = max(tiempos)
    
    # Comparación de alturas
    mejor_que_hr = sum(1 for h_p, h_hr in zip(alturas_pointer, alturas_hr) if h_p < h_hr)
    igual_que_hr = sum(1 for h_p, h_hr in zip(alturas_pointer, alturas_hr) if h_p == h_hr)
    peor_que_hr = sum(1 for h_p, h_hr in zip(alturas_pointer, alturas_hr) if h_p > h_hr)
    
    print(f"\nDESEMPEÑO GENERAL:")
    print(f"  Tasa de éxito: {exitos}/{n_problemas_test} ({tasa_exito:.1f}%)")
    print(f"  Tiempo promedio: {tiempo_promedio:.2f} ms")
    print(f"  Tiempo min/max: {tiempo_min:.2f} / {tiempo_max:.2f} ms")
    
    print(f"\nCOMPARACIÓN CON HR PURO:")
    print(f"  Mejor que HR: {mejor_que_hr}/{n_problemas_test} ({mejor_que_hr/n_problemas_test*100:.1f}%)")
    print(f"  Igual que HR: {igual_que_hr}/{n_problemas_test} ({igual_que_hr/n_problemas_test*100:.1f}%)")
    print(f"  Peor que HR: {peor_que_hr}/{n_problemas_test} ({peor_que_hr/n_problemas_test*100:.1f}%)")
    
    # Altura promedio
    altura_prom_hr = sum(alturas_hr) / len(alturas_hr)
    altura_prom_pointer = sum(alturas_pointer) / len(alturas_pointer)
    diff_prom = altura_prom_pointer - altura_prom_hr
    perc_diff = (diff_prom / altura_prom_hr * 100) if altura_prom_hr > 0 else 0
    
    print(f"\nALTURA PROMEDIO:")
    print(f"  HR Puro: {altura_prom_hr:.2f}")
    print(f"  Pointer: {altura_prom_pointer:.2f}")
    print(f"  Diferencia: {diff_prom:+.2f} ({perc_diff:+.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"TEST SUITE COMPLETADO")
    print(f"{'='*80}\n")
    
    return {
        'tasa_exito': tasa_exito,
        'tiempo_promedio': tiempo_promedio,
        'mejor_que_hr': mejor_que_hr,
        'igual_que_hr': igual_que_hr,
        'peor_que_hr': peor_que_hr,
        'altura_prom_hr': altura_prom_hr,
        'altura_prom_pointer': altura_prom_pointer
    }


def test_comparacion_completa(
    model_path: str = None,
    n_problemas: int = 10,
    categoria: str = "C1",
    device: str = None,
    incluir_greedy: bool = True
):
    """Test completo comparando Pointer con HR y algoritmos greedy.
    
    Ejecuta múltiples problemas y genera estadísticas comparativas
    entre todos los algoritmos.
    
    Args:
        model_path: Ruta al modelo .pth (o None para auto-detectar)
        n_problemas: Número de problemas a testear
        categoria: Categoría del problema
        device: 'cpu' o 'cuda' (None = auto-detectar)
        incluir_greedy: Si True, incluye FFDH, BFDH, NF, BL
    """
    print(f"\n{'='*80}")
    print(f"TEST COMPARACIÓN COMPLETA")
    print(f"{'='*80}")
    
    # Auto-detectar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargar modelo
    if model_path is None:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) 
                          if f.startswith(f"pointer_{categoria.lower()}") and f.endswith(".pth")]
            if model_files:
                model_files.sort(reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
                print(f"Auto-detectado: {model_path}")
    
    model, _ = cargar_modelo_entrenado(model_path, device)
    
    # Generar problemas
    print(f"\nGenerando {n_problemas} problemas de test...")
    problems, W, H = gen.generate_problems_guillotine(categoria, n_problemas, export=False)
    print(f"{len(problems)} problemas generados")
    
    # Inicializar estadísticas por algoritmo
    stats = {}
    algoritmos = ['hr', 'pointer']
    if incluir_greedy:
        algoritmos.extend(['ffdh', 'bfdh', 'nf', 'bl'])
    
    for alg in algoritmos:
        stats[alg] = {
            'alturas': [],
            'tiempos': [],
            'exitos': 0
        }
    
    # Procesar cada problema
    print(f"\n{'='*80}")
    print(f"EJECUTANDO TESTS")
    print(f"{'='*80}")
    
    for idx, rects in enumerate(problems):
        print(f"\nProblema {idx+1}/{n_problemas}:")
        
        # HR
        start = time.time()
        _, altura_hr, _, _ = hr.heuristic_recursion(rects, W, categoria)
        tiempo_hr = (time.time() - start) * 1000
        stats['hr']['alturas'].append(altura_hr)
        stats['hr']['tiempos'].append(tiempo_hr)
        stats['hr']['exitos'] += 1
        
        # Pointer
        start = time.time()
        _, altura_pointer, _, _ = hrp.heuristic_recursion_pointer(
            rects, W, categoria, model, device
        )
        tiempo_pointer = (time.time() - start) * 1000
        stats['pointer']['alturas'].append(altura_pointer)
        stats['pointer']['tiempos'].append(tiempo_pointer)
        stats['pointer']['exitos'] += 1
        
        # Greedy algorithms
        if incluir_greedy:
            resultados_greedy = greedy.comparar_algoritmos(rects, W)
            for alg_key in ['ffdh', 'bfdh', 'nf', 'bl']:
                stats[alg_key]['alturas'].append(resultados_greedy[alg_key]['altura'])
                stats[alg_key]['tiempos'].append(resultados_greedy[alg_key]['tiempo_ms'])
                stats[alg_key]['exitos'] += 1
        
        # Mostrar resumen compacto
        print(f"  HR: {altura_hr:.1f} ({tiempo_hr:.2f}ms) | Pointer: {altura_pointer:.1f} ({tiempo_pointer:.2f}ms)", end="")
        if incluir_greedy:
            print(f" | FFDH: {resultados_greedy['ffdh']['altura']:.1f}")
        else:
            print()
    
    # ============================================================
    # ESTADÍSTICAS FINALES
    # ============================================================
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS COMPARATIVAS")
    print(f"{'='*80}")
    
    # Tabla de promedios
    print(f"\n{'Algoritmo':<30} {'Altura Prom':<15} {'Tiempo Prom':<15} {'vs HR':<15}")
    print(f"{'-'*80}")
    
    altura_prom_hr = sum(stats['hr']['alturas']) / len(stats['hr']['alturas'])
    
    nombres_alg = {
        'hr': 'HR (Heuristic Recursion)',
        'pointer': 'Pointer Model',
        'ffdh': 'FFDH',
        'bfdh': 'BFDH',
        'nf': 'Next Fit',
        'bl': 'Bottom-Left'
    }
    
    for alg in algoritmos:
        if not stats[alg]['alturas']:
            continue
            
        altura_prom = sum(stats[alg]['alturas']) / len(stats[alg]['alturas'])
        tiempo_prom = sum(stats[alg]['tiempos']) / len(stats[alg]['tiempos'])
        diff_vs_hr = altura_prom - altura_prom_hr
        perc_vs_hr = (diff_vs_hr / altura_prom_hr * 100) if altura_prom_hr > 0 else 0
        
        print(f"{nombres_alg[alg]:<30} {altura_prom:<15.2f} {tiempo_prom:<15.3f} {diff_vs_hr:+.2f} ({perc_vs_hr:+.1f}%)")
    
    # Análisis detallado Pointer vs HR
    print(f"\n{'='*80}")
    print(f"ANÁLISIS POINTER vs HR")
    print(f"{'='*80}")
    
    mejor_que_hr = sum(1 for h_p, h_hr in zip(stats['pointer']['alturas'], stats['hr']['alturas']) if h_p < h_hr)
    igual_que_hr = sum(1 for h_p, h_hr in zip(stats['pointer']['alturas'], stats['hr']['alturas']) if h_p == h_hr)
    peor_que_hr = sum(1 for h_p, h_hr in zip(stats['pointer']['alturas'], stats['hr']['alturas']) if h_p > h_hr)
    
    print(f"Mejor que HR: {mejor_que_hr}/{n_problemas} ({mejor_que_hr/n_problemas*100:.1f}%)")
    print(f"Igual que HR: {igual_que_hr}/{n_problemas} ({igual_que_hr/n_problemas*100:.1f}%)")
    print(f"Peor que HR: {peor_que_hr}/{n_problemas} ({peor_que_hr/n_problemas*100:.1f}%)")
    
    altura_prom_pointer = sum(stats['pointer']['alturas']) / len(stats['pointer']['alturas'])
    tiempo_prom_pointer = sum(stats['pointer']['tiempos']) / len(stats['pointer']['tiempos'])
    tiempo_prom_hr = sum(stats['hr']['tiempos']) / len(stats['hr']['tiempos'])
    speedup = tiempo_prom_hr / tiempo_prom_pointer if tiempo_prom_pointer > 0 else 0
    
    print(f"\nSpeedup: {speedup:.2f}x {'(más rápido)' if speedup > 1 else '(más lento)'}")
    
    print(f"\n{'='*80}")
    print(f"TEST COMPLETADO")
    print(f"{'='*80}\n")
    
    return stats


# ============================================================
# FUNCIÓN PRINCIPAL PARA LLAMAR DESDE MAIN
# ============================================================
def test_modelo_pointer(
    model_path: str = None,
    n_problemas: int = 10,
    categoria: str = "C1",
    device: str = None,
    comparar_hr: bool = True
):
    """Función principal de testing para llamar desde main.py o scripts.
    
    Args:
        model_path: Ruta al modelo .pth. Si None, busca el último modelo en models/
        n_problemas: Número de problemas de test
        categoria: Categoría de problemas
        device: 'cpu' o 'cuda'. Si None, detecta automáticamente
        comparar_hr: Si comparar con HR puro
        
    Returns:
        Dict con estadísticas del test
    """
    # Auto-detectar device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Auto-detectar último modelo si no se especifica
    if model_path is None:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.startswith(f"pointer_{categoria.lower()}") and f.endswith(".pth")]
            if model_files:
                # Ordenar por accuracy (está en el nombre)
                model_files.sort(reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
                print(f"Auto-detectado modelo: {model_path}")
            else:
                raise FileNotFoundError(f"No se encontraron modelos en {models_dir}/ para categoría {categoria}")
        else:
            raise FileNotFoundError(f"Directorio {models_dir}/ no existe")
    
    # Ejecutar test suite
    return test_suite_completo(model_path, n_problemas, categoria, device)


def test_y_visualizar_problema(
    model_path: str,
    problema_rects: Optional[List[Tuple[int, int]]] = None,
    categoria: str = "C1",
    device: str = "cpu",
    comparar_hr: bool = True
):
    """Testea y visualiza un problema específico.
    
    Args:
        model_path: Ruta al modelo .pth
        problema_rects: Lista de rectángulos. Si None, genera uno aleatorio
        categoria: Categoría del problema
        device: 'cpu' o 'cuda'
        comparar_hr: Si True, muestra comparación lado a lado con HR
    """
    print(f"\n{'='*80}")
    print(f"TEST Y VISUALIZACIÓN DE PROBLEMA INDIVIDUAL")
    print(f"{'='*80}")
    
    # Cargar modelo
    model, _ = cargar_modelo_entrenado(model_path, device)
    
    # Generar problema si no se proporciona
    if problema_rects is None:
        problems, _, _ = gen.generate_problems_guillotine(categoria, 1, export=False)
        problema_rects = problems[0]
    
    print(f"\nProblema: {len(problema_rects)} rectángulos")
    
    # Ejecutar pointer model
    W = cat.CATEGORIES[categoria]["width"]
    H = cat.CATEGORIES[categoria]["height"]
    
    print(f"\nEjecutando Pointer Model...")
    start = time.time()
    placements_pointer, altura_pointer, rect_seq_pointer, _ = hrp.heuristic_recursion_pointer(
        rects=problema_rects,
        container_width=W,
        model=model,
        category=categoria,
        device=device
    )
    tiempo_pointer_ms = (time.time() - start) * 1000
    
    print(f"  Altura: {altura_pointer}")
    print(f"  Tiempo: {tiempo_pointer_ms:.2f} ms")
    print(f"  Rectángulos colocados: {len(placements_pointer)}/{len(problema_rects)}")
    
    if comparar_hr:
        # Ejecutar HR puro
        print(f"\nEjecutando HR Puro...")
        start = time.time()
        placements_hr, altura_hr, _, _ = hr.heuristic_recursion(
            rects=problema_rects.copy(),
            container_width=W,
            category=categoria
        )
        tiempo_hr_ms = (time.time() - start) * 1000
        
        print(f"  Altura: {altura_hr}")
        print(f"  Tiempo: {tiempo_hr_ms:.2f} ms")
        print(f"  Rectángulos colocados: {len(placements_hr)}/{len(problema_rects)}")
        
        # Visualización comparativa
        print(f"\nGenerando visualización comparativa...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # HR Puro (izquierda)
        _dibujar_packing_en_axes(
            ax=ax1,
            placements=placements_hr,
            container_width=W,
            container_height=H,
            titulo=f"HR Puro\nAltura: {altura_hr} | Tiempo: {tiempo_hr_ms:.1f}ms"
        )
        
        # Pointer Model (derecha)
        _dibujar_packing_en_axes(
            ax=ax2,
            placements=placements_pointer,
            container_width=W,
            container_height=H,
            titulo=f"Pointer Model\nAltura: {altura_pointer} | Tiempo: {tiempo_pointer_ms:.1f}ms"
        )
        
        # Título general con diferencia
        diff = altura_pointer - altura_hr
        diff_str = f"{diff:+}" if diff != 0 else "IGUAL"
        fig.suptitle(
            f"Comparación: {len(problema_rects)} rectángulos | Diferencia altura: {diff_str}",
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Solo visualizar pointer model
        print(f"\nGenerando visualización...")
        fig = hr.visualizar_packing(placements_pointer, W, H, show=True)
        fig.axes[0].set_title(
            f"Pointer Model - {len(problema_rects)} rectángulos\nAltura: {altura_pointer} | Tiempo: {tiempo_pointer_ms:.1f}ms",
            fontsize=12,
            fontweight='bold'
        )


def comparar_todos_algoritmos(
    problema_rects: List[Tuple[int, int]],
    model: SPPPointerModel,
    categoria: str,
    device: str = "cpu",
    incluir_greedy: bool = True,
    visualizar: bool = False
) -> dict:
    """Compara Pointer Model con HR y algoritmos greedy clásicos.
    
    Args:
        problema_rects: Lista de rectángulos a empaquetar
        model: Modelo pointer entrenado
        categoria: Categoría del problema
        device: 'cpu' o 'cuda'
        incluir_greedy: Si True, incluye FFDH, BFDH, NF, BL
        visualizar: Si True, muestra gráfico comparativo
        
    Returns:
        Dict con resultados de cada algoritmo
    """
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN COMPLETA: POINTER vs HR vs GREEDY ALGORITHMS")
    print(f"{'='*80}")
    
    W = cat.CATEGORIES[categoria]["width"]
    H = cat.CATEGORIES[categoria]["height"]
    resultados = {}
    
    # ============ HR PURO ============
    print(f"\nEjecutando HR Puro...")
    start = time.time()
    placements_hr, altura_hr, _, _ = hr.heuristic_recursion(
        rects=problema_rects.copy(),
        container_width=W,
        category=categoria
    )
    tiempo_hr = (time.time() - start) * 1000
    
    resultados['hr'] = {
        'nombre': 'HR (Heuristic Recursion)',
        'placements': placements_hr,
        'altura': altura_hr,
        'tiempo_ms': tiempo_hr,
        'n_colocados': len(placements_hr)
    }
    
    # ============ POINTER MODEL ============
    print(f"Ejecutando Pointer Model...")
    start = time.time()
    placements_pointer, altura_pointer, _, _ = hrp.heuristic_recursion_pointer(
        rects=problema_rects.copy(),
        container_width=W,
        model=model,
        category=categoria,
        device=device
    )
    tiempo_pointer = (time.time() - start) * 1000
    
    resultados['pointer'] = {
        'nombre': 'Pointer Model',
        'placements': placements_pointer,
        'altura': altura_pointer,
        'tiempo_ms': tiempo_pointer,
        'n_colocados': len(placements_pointer)
    }
    
    # ============ ALGORITMOS GREEDY ============
    if incluir_greedy:
        print(f"Ejecutando algoritmos greedy clásicos...")
        resultados_greedy = greedy.comparar_algoritmos(
            problema_rects, 
            W, 
            algoritmos=['ffdh', 'bfdh', 'nf', 'bl']
        )
        resultados.update(resultados_greedy)
    
    # ============ TABLA COMPARATIVA ============
    print(f"\n{'='*80}")
    print(f"TABLA COMPARATIVA")
    print(f"{'='*80}")
    print(f"\n{'Algoritmo':<35} {'Altura':<12} {'Tiempo (ms)':<15} {'vs HR altura':<15}")
    print(f"{'-'*80}")
    
    # Ordenar por altura (mejor primero)
    orden = ['nf', 'bl', 'bfdh', 'ffdh', 'hr', 'pointer']
    
    for alg_key in orden:
        if alg_key not in resultados:
            continue
            
        res = resultados[alg_key]
        diff_vs_hr = res['altura'] - altura_hr
        perc_vs_hr = (diff_vs_hr / altura_hr * 100) if altura_hr > 0 else 0
        
        print(f"{res['nombre']:<35} {res['altura']:<12.1f} {res['tiempo_ms']:<15.3f} {diff_vs_hr:+.1f} ({perc_vs_hr:+.1f}%)")
    
    # ============ ANÁLISIS ============
    print(f"\n{'='*80}")
    print(f"ANÁLISIS")
    print(f"{'='*80}")
    
    # Mejor altura
    mejor_altura = min(res['altura'] for res in resultados.values())
    mejor_alg = [k for k, v in resultados.items() if v['altura'] == mejor_altura][0]
    print(f"\nMejor altura: {resultados[mejor_alg]['nombre']} ({mejor_altura})")
    
    # Más rápido
    mas_rapido = min(res['tiempo_ms'] for res in resultados.values())
    mas_rapido_alg = [k for k, v in resultados.items() if v['tiempo_ms'] == mas_rapido][0]
    print(f"Más rápido: {resultados[mas_rapido_alg]['nombre']} ({mas_rapido:.3f}ms)")
    
    # Análisis Pointer vs HR
    diff_altura_pointer_hr = altura_pointer - altura_hr
    perc_altura = (diff_altura_pointer_hr / altura_hr * 100) if altura_hr > 0 else 0
    speedup = tiempo_hr / tiempo_pointer if tiempo_pointer > 0 else 0
    
    print(f"\nPointer vs HR:")
    print(f"  Altura: {diff_altura_pointer_hr:+.1f} ({perc_altura:+.1f}%)")
    print(f"  Speedup: {speedup:.1f}x {'(más rápido)' if speedup > 1 else '(más lento)'}")
    
    # Análisis Pointer vs FFDH (baseline común)
    if 'ffdh' in resultados:
        diff_pointer_ffdh = altura_pointer - resultados['ffdh']['altura']
        perc_ffdh = (diff_pointer_ffdh / resultados['ffdh']['altura'] * 100)
        print(f"\nPointer vs FFDH (baseline):")
        print(f"  Altura: {diff_pointer_ffdh:+.1f} ({perc_ffdh:+.1f}%)")
    
    # ============ VISUALIZACIÓN ============
    if visualizar and incluir_greedy:
        print(f"\nGenerando visualización comparativa...")
        
        # Seleccionar 4 algoritmos representativos: FFDH, HR, Pointer, BFDH
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        algoritmos_viz = [
            ('ffdh', 0),
            ('bfdh', 1),
            ('hr', 2),
            ('pointer', 3)
        ]
        
        for alg_key, idx in algoritmos_viz:
            if alg_key not in resultados:
                continue
                
            res = resultados[alg_key]
            _dibujar_packing_en_axes(
                ax=axes[idx],
                placements=res['placements'],
                container_width=W,
                container_height=H,
                titulo=f"{res['nombre']}\nAltura: {res['altura']:.1f} | Tiempo: {res['tiempo_ms']:.2f}ms"
            )
        
        plt.tight_layout()
        plt.show()
    
    return resultados


def visualizar_mejores_peores_casos(
    model_path: str,
    n_problemas: int = 20,
    n_mostrar: int = 3,
    categoria: str = "C1",
    device: str = "cpu"
):
    """Ejecuta tests y muestra los mejores y peores casos del modelo vs HR.
    
    Args:
        model_path: Ruta al modelo .pth
        n_problemas: Número de problemas a testear
        n_mostrar: Cuántos mejores/peores casos mostrar
        categoria: Categoría del problema
        device: 'cpu' o 'cuda'
    """
    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE MEJORES Y PEORES CASOS")
    print(f"{'='*80}")
    
    # Cargar modelo
    model, _ = cargar_modelo_entrenado(model_path, device)
    
    # Generar problemas
    print(f"\nGenerando {n_problemas} problemas...")
    problems, W, H = gen.generate_problems_guillotine(categoria, n_problemas, export=False)
    
    # Ejecutar tests
    print(f"Ejecutando tests...")
    resultados = []
    
    for idx, rects in enumerate(problems):
        print(f"  Problema {idx+1}/{n_problemas}...", end='\r')
        
        # Pointer
        _, altura_pointer, _, _ = hrp.heuristic_recursion_pointer(
            rects=rects.copy(),
            container_width=W,
            model=model,
            category=categoria,
            device=device
        )
        
        # HR
        placements_hr, altura_hr, _, _ = hr.heuristic_recursion(
            rects=rects.copy(),
            container_width=W,
            category=categoria
        )
        
        diff = altura_pointer - altura_hr
        
        resultados.append({
            'idx': idx,
            'rects': rects,
            'altura_pointer': altura_pointer,
            'altura_hr': altura_hr,
            'diff': diff,
            'placements_hr': placements_hr,
            'placements_pointer': None  # Se genera cuando se necesite
        })
    
    print(f"\nTests completados")
    
    # Ordenar por diferencia
    resultados.sort(key=lambda x: x['diff'])
    
    # Mejores casos (pointer mejor que HR)
    mejores = [r for r in resultados if r['diff'] < 0][:n_mostrar]
    # Peores casos (pointer peor que HR)
    peores = [r for r in resultados if r['diff'] > 0][-n_mostrar:]
    peores.reverse()
    
    # Visualizar mejores casos
    if mejores:
        print(f"\n{'='*80}")
        print(f"TOP {len(mejores)} MEJORES CASOS (Pointer supera a HR)")
        print(f"{'='*80}")
        
        for i, res in enumerate(mejores):
            print(f"\n--- Caso {i+1}: Problema #{res['idx']+1} ---")
            print(f"  Altura Pointer: {res['altura_pointer']}")
            print(f"  Altura HR: {res['altura_hr']}")
            print(f"  Mejora: {abs(res['diff'])} ({abs(res['diff'])/res['altura_hr']*100:.1f}%)")
            
            # Re-generar placements pointer
            placements_pointer, _, _, _ = hrp.heuristic_recursion_pointer(
                rects=res['rects'].copy(),
                container_width=W,
                model=model,
                category=categoria,
                device=device
            )
            
            # Visualizar
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # HR Puro (izquierda)
            _dibujar_packing_en_axes(
                ax=ax1,
                placements=res['placements_hr'],
                container_width=W,
                container_height=H,
                titulo=f"HR Puro - Altura: {res['altura_hr']}"
            )
            
            # Pointer Model (derecha)
            _dibujar_packing_en_axes(
                ax=ax2,
                placements=placements_pointer,
                container_width=W,
                container_height=H,
                titulo=f"Pointer Model - Altura: {res['altura_pointer']}"
            )
            
            fig.suptitle(
                f"Mejor Caso #{i+1} - Mejora: {abs(res['diff'])} ({abs(res['diff'])/res['altura_hr']*100:.1f}%)",
                fontsize=14,
                fontweight='bold',
                color='green'
            )
            
            plt.tight_layout()
            plt.show()
    
    # Visualizar peores casos
    if peores:
        print(f"\n{'='*80}")
        print(f"TOP {len(peores)} PEORES CASOS (Pointer peor que HR)")
        print(f"{'='*80}")
        
        for i, res in enumerate(peores):
            print(f"\n--- Caso {i+1}: Problema #{res['idx']+1} ---")
            print(f"  Altura Pointer: {res['altura_pointer']}")
            print(f"  Altura HR: {res['altura_hr']}")
            print(f"  Diferencia: +{res['diff']} (+{res['diff']/res['altura_hr']*100:.1f}%)")
            
            # Re-generar placements pointer
            placements_pointer, _, _, _ = hrp.heuristic_recursion_pointer(
                rects=res['rects'].copy(),
                container_width=W,
                model=model,
                category=categoria,
                device=device
            )
            
            # Visualizar
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # HR Puro (izquierda)
            _dibujar_packing_en_axes(
                ax=ax1,
                placements=res['placements_hr'],
                container_width=W,
                container_height=H,
                titulo=f"HR Puro - Altura: {res['altura_hr']}"
            )
            
            # Pointer Model (derecha)
            _dibujar_packing_en_axes(
                ax=ax2,
                placements=placements_pointer,
                container_width=W,
                container_height=H,
                titulo=f"Pointer Model - Altura: {res['altura_pointer']}"
            )
            
            fig.suptitle(
                f"Peor Caso #{i+1} - Diferencia: +{res['diff']} (+{res['diff']/res['altura_hr']*100:.1f}%)",
                fontsize=14,
                fontweight='bold',
                color='red'
            )
            
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Ejemplo de uso standalone
    print("Iniciando testing del modelo pointer...")
    
    # Configuración
    CATEGORIA = "C1"
    N_PROBLEMAS_TEST = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Buscar último modelo entrenado
    MODEL_PATH = "models/pointer_c1_L2_H16_acc8235.pth"  # Auto-detect
    
    # Ejecutar tests
    resultados = test_modelo_pointer(
        model_path=MODEL_PATH,
        n_problemas=N_PROBLEMAS_TEST,
        categoria=CATEGORIA,
        device=DEVICE
    )
    
    print(f"\nTesting completado con éxito!")
