"""
Análisis de complejidad temporal y espacial de algoritmos de empaquetamiento.

Genera:
1. Boxplots de distribución de alturas por algoritmo y categoría.
2. Tablas estadísticas con métricas de desempeño.
3. Análisis de complejidad teórica y medida.
"""
import os
import sys
import time
import tracemalloc
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import psutil

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

import generator as gen
import categories as cat
import hr_algorithm as hr
import hr_pointer as hrp
import greedy_algorithms as greedy
from pointer_model import SPPPointerModel
from test_pointer import cargar_modelo_entrenado


# Fix para el error de OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


@dataclass
class ResultadoAlgoritmo:
    algoritmo: str
    categoria: str
    problema_idx: int
    altura: float
    tiempo_ms: float
    memoria_mb: float
    num_rectangulos: int
    
    
@dataclass
class ComplejidadTeorica:
    algoritmo: str
    temporal_big_o: str
    temporal_descripcion: str
    espacial_big_o: str
    espacial_descripcion: str
    
    
# ============================================================
# COMPLEJIDAD TEÓRICA DE CADA ALGORITMO
# ============================================================
COMPLEJIDADES = {
    "HR": ComplejidadTeorica(
        algoritmo="HR (Heuristic Recursion)",
        temporal_big_o="O(N²)",
        temporal_descripcion="Prueba permutaciones de N rectángulos, cada empaquetamiento es O(N²) por búsqueda de espacios",
        espacial_big_o="O(N × S)",
        espacial_descripcion="N rectángulos + S espacios activos (típicamente S ≤ 2N)"
    ),
    "Pointer": ComplejidadTeorica(
        algoritmo="Pointer Model",
        temporal_big_o="O(N² × d²)",
        temporal_descripcion="N decisiones × complejidad Transformer (d=dimensión modelo, típicamente d²≈65k ops)",
        espacial_big_o="O(N²)",
        espacial_descripcion="Encoder cacheado (N×d) + parámetros del modelo (P≈2M parámetros × 4 bytes)"
    ),
    "FFDH": ComplejidadTeorica(
        algoritmo="FFDH (First Fit Decreasing Height)",
        temporal_big_o="O(N log N)",
        temporal_descripcion="Ordenar por altura O(N log N) + colocar N rects con búsqueda O(N) niveles",
        espacial_big_o="O(N)",
        espacial_descripcion="Lista de rectángulos + niveles activos (típicamente < N/2)"
    ),
    "BFDH": ComplejidadTeorica(
        algoritmo="BFDH (Best Fit Decreasing Height)",
        temporal_big_o="O(N log N)",
        temporal_descripcion="Ordenar por altura O(N log N) + colocar N rects buscando mejor nivel O(N)",
        espacial_big_o="O(N)",
        espacial_descripcion="Lista de rectángulos + niveles activos"
    ),
    "Next Fit": ComplejidadTeorica(
        algoritmo="Next Fit",
        temporal_big_o="O(N)",
        temporal_descripcion="Colocación lineal sin backtracking (un solo nivel activo)",
        espacial_big_o="O(N)",
        espacial_descripcion="Lista de rectángulos + nivel actual"
    ),
    "Bottom-Left": ComplejidadTeorica(
        algoritmo="Bottom-Left",
        temporal_big_o="O(N²)",
        temporal_descripcion="N rectángulos × búsqueda de posición más abajo-izquierda O(N) comparaciones",
        espacial_big_o="O(N)",
        espacial_descripcion="Lista de rectángulos colocados"
    )
}


# ============================================================
# MEDICIÓN DE TIEMPO Y MEMORIA
# ============================================================
def medir_tiempo_memoria(func, *args, **kwargs) -> Tuple[any, float, float]:
    """Mide tiempo de ejecución y uso de memoria de una función."""

    # Limpiar memoria antes de medir
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Medir memoria inicial
    tracemalloc.start()
    proceso = psutil.Process()
    mem_antes = proceso.memory_info().rss / 1024 / 1024  # MB
    
    # Medir tiempo
    inicio = time.perf_counter()
    resultado = func(*args, **kwargs)
    fin = time.perf_counter()
    
    # Medir memoria final
    mem_despues = proceso.memory_info().rss / 1024 / 1024  # MB
    _, pico_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tiempo_ms = (fin - inicio) * 1000
    memoria_mb = max(mem_despues - mem_antes, pico_mem / 1024 / 1024)
    
    return resultado, tiempo_ms, memoria_mb


# ============================================================
# EJECUTAR ALGORITMOS
# ============================================================
def ejecutar_hr(rects: List[Tuple[int, int]], W: int, categoria: str) -> Tuple[List, float, float]:
    """Ejecuta HR"""
    def run():
        placements, altura, _, _ = hr.heuristic_recursion(rects, W, categoria)
        return placements, altura
    
    (placements, altura), tiempo_ms, memoria_mb = medir_tiempo_memoria(run)
    return placements, altura, tiempo_ms, memoria_mb


def ejecutar_pointer(rects: List[Tuple[int, int]], W: int, categoria: str, 
                     model: SPPPointerModel, device: str) -> Tuple[List, float, float]:
    """Ejecuta Pointer Model"""
    def run():
        placements, altura, _, _ = hrp.heuristic_recursion_pointer(
            rects=rects.copy(),
            container_width=W,
            category=categoria,
            model=model,
            device=device
        )
        # Convertir altura a float si es tensor
        if isinstance(altura, torch.Tensor):
            altura = float(altura.item())
        return placements, altura
    
    (placements, altura), tiempo_ms, memoria_mb = medir_tiempo_memoria(run)
    return placements, altura, tiempo_ms, memoria_mb


def ejecutar_greedy(rects: List[Tuple[int, int]], W: int, algoritmo: str) -> Tuple[List, float, float]:
    """Ejecuta algoritmo greedy"""
    def run():
        if algoritmo == "FFDH":
            placements, altura = greedy.ffdh_packing(rects.copy(), W)
        elif algoritmo == "BFDH":
            placements, altura = greedy.bfdh_packing(rects.copy(), W)
        elif algoritmo == "Next Fit":
            placements, altura = greedy.next_fit_packing(rects.copy(), W)
        elif algoritmo == "Bottom-Left":
            placements, altura = greedy.bottom_left_packing(rects.copy(), W)
        else:
            raise ValueError(f"Algoritmo desconocido: {algoritmo}")
        return placements, altura
    
    (placements, altura), tiempo_ms, memoria_mb = medir_tiempo_memoria(run)
    return placements, altura, tiempo_ms, memoria_mb


# ============================================================
# GENERAR RESULTADOS PARA UNA CATEGORÍA
# ============================================================
def analizar_categoria(
    categoria: str,
    n_problemas: int,
    model_path: Optional[str] = None,
    device: str = "cpu",
    save_mejor_casos: bool = True,
    output_dir: str = "img/complexity_analysis"
) -> List[ResultadoAlgoritmo]:
    """Ejecuta todos los algoritmos en n_problemas de una categoría.
    
    Returns:
        Lista de ResultadoAlgoritmo con todas las ejecuciones
    """
    print(f"\n{'='*80}")
    print(f"ANALIZANDO CATEGORÍA: {categoria}")
    print(f"{'='*80}")
    
    W = cat.CATEGORIES[categoria]["width"]
    Href = cat.CATEGORIES[categoria]["height"]
    N = cat.CATEGORIES[categoria]["num_items"]
    
    # Cargar modelo Pointer si existe
    model = None
    categoria_modelo = None
    if model_path or os.path.exists("models"):
        try:
            model, categoria_modelo = cargar_modelo_entrenado(model_path, device)
            print(f"✓ Modelo Pointer cargado")
        except Exception as e:
            print(f"✗ No se pudo cargar modelo Pointer: {e}")
    
    # Generar problemas
    print(f"\nGenerando {n_problemas} problemas...")
    problemas, _, _ = gen.generate_problems_guillotine(categoria, n_problemas)
    print(f"✓ {len(problemas)} problemas generados")
    
    if len(problemas) < n_problemas:
        print(f"⚠ Advertencia: Solo se generaron {len(problemas)}/{n_problemas} problemas válidos")
    
    resultados = []
    mejores_casos = []  # Lista de (idx, rects, altura_hr, placements_hr, altura_ptr, placements_ptr)
    casos_empate = 0  # Contador de casos donde Pointer == HR
    algoritmos = ["HR", "FFDH", "BFDH", "Next Fit", "Bottom-Left"]
    if model:
        algoritmos.insert(1, "Pointer")  # Insertar después de HR
    
    print(f"\nEjecutando algoritmos: {', '.join(algoritmos)}")
    print(f"{'─'*80}")
    
    for idx, rects in enumerate(problemas, 1):
        altura_hr = None
        placements_hr = None
        altura_ptr = None
        placements_ptr = None
        print(f"Problema {idx}/{len(problemas)}: ", end="", flush=True)
        
        # HR
        try:
            placements_hr, altura_hr, tiempo_hr, memoria_hr = ejecutar_hr(rects.copy(), W, categoria)
            resultados.append(ResultadoAlgoritmo(
                algoritmo="HR",
                categoria=categoria,
                problema_idx=idx,
                altura=altura_hr,
                tiempo_ms=tiempo_hr,
                memoria_mb=memoria_hr,
                num_rectangulos=N
            ))
            print(f"HR:{altura_hr:.0f} ", end="", flush=True)
        except Exception as e:
            if idx == 1:  # Solo imprimir error detallado del primer problema
                print(f"\n  HR Error detallado: {e}")
            print(f"HR:ERROR ", end="", flush=True)
        
        # Pointer
        if model:
            try:
                placements_ptr, altura_ptr, tiempo_ptr, memoria_ptr = ejecutar_pointer(
                    rects.copy(), W, categoria, model, device
                )
                resultados.append(ResultadoAlgoritmo(
                    algoritmo="Pointer",
                    categoria=categoria,
                    problema_idx=idx,
                    altura=altura_ptr,
                    tiempo_ms=tiempo_ptr,
                    memoria_mb=memoria_ptr,
                    num_rectangulos=N
                ))
                print(f"Ptr:{altura_ptr:.0f} ", end="", flush=True)
                
                # Detectar si Pointer superó a HR
                if altura_hr is not None and altura_ptr < altura_hr:
                    mejores_casos.append((idx, rects.copy(), altura_hr, placements_hr, altura_ptr, placements_ptr))
                    print(f"★ ", end="", flush=True)  # Marcar caso especial
                # Detectar si Pointer empató con HR
                elif altura_hr is not None and abs(altura_ptr - altura_hr) < 0.01:  # Empate (tolerancia para floats)
                    casos_empate += 1
                    print(f"= ", end="", flush=True)  # Marcar empate
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')[:20] if str(e) else "unknown"
                print(f"Ptr:ERROR({error_msg}) ", end="", flush=True)
        
        # Greedy algorithms
        for alg in ["FFDH", "BFDH", "Next Fit", "Bottom-Left"]:
            try:
                _, altura, tiempo, memoria = ejecutar_greedy(rects.copy(), W, alg)
                resultados.append(ResultadoAlgoritmo(
                    algoritmo=alg,
                    categoria=categoria,
                    problema_idx=idx,
                    altura=altura,
                    tiempo_ms=tiempo,
                    memoria_mb=memoria,
                    num_rectangulos=N
                ))
                print(f"{alg[:2]}:{altura:.0f} ", end="", flush=True)
            except Exception as e:
                print(f"{alg[:2]}:ERROR({str(e)[:15]}) ", end="", flush=True)
        
        print()  # Nueva línea
    
    print(f"\n✓ Análisis completado: {len(resultados)} ejecuciones")
    
    # Calcular estadísticas de superioridad Pointer vs HR
    if model:
        total_comparaciones = len([r for r in resultados if r.algoritmo in ["HR", "Pointer"]])
        casos_pointer_mejor = len(mejores_casos)
        casos_totales = total_comparaciones // 2
        casos_hr_mejor = casos_totales - casos_pointer_mejor - casos_empate
        
        if casos_totales > 0:
            porcentaje_pointer_mejor = (casos_pointer_mejor / casos_totales) * 100
            porcentaje_empate = (casos_empate / casos_totales) * 100
            porcentaje_hr_mejor = (casos_hr_mejor / casos_totales) * 100
            
            print(f"  • Pointer mejor que HR: {casos_pointer_mejor}/{casos_totales} casos ({porcentaje_pointer_mejor:.1f}%)")
            print(f"  • Pointer igual a HR: {casos_empate}/{casos_totales} casos ({porcentaje_empate:.1f}%)")
            print(f"  • HR mejor que Pointer: {casos_hr_mejor}/{casos_totales} casos ({porcentaje_hr_mejor:.1f}%)")
    
    # Guardar mejores casos donde Pointer superó a HR
    if save_mejor_casos and mejores_casos:
        print(f"\n★ Pointer superó a HR en {len(mejores_casos)} casos")
        os.makedirs(output_dir, exist_ok=True)
        mejores_dir = os.path.join(output_dir, f"mejores_{categoria}")
        os.makedirs(mejores_dir, exist_ok=True)
        
        for idx_caso, (problema_idx, rects, h_hr, plc_hr, h_ptr, plc_ptr) in enumerate(mejores_casos[:10], 1):
            diff = h_hr - h_ptr
            filename = f"mejor_{categoria}_p{problema_idx}_diff{diff:.0f}.png"
            filepath = os.path.join(mejores_dir, filename)
            
            # Guardar el problema original como texto
            problema_txt = os.path.join(mejores_dir, f"mejor_{categoria}_p{problema_idx}_problema.txt")
            with open(problema_txt, 'w') as f:
                f.write(f"Categoría: {categoria}\n")
                f.write(f"Problema #{problema_idx}\n")
                f.write(f"Contenedor Width: {W}\n")
                f.write(f"Número de rectángulos: {len(rects)}\n")
                f.write(f"\nHR Altura: {h_hr:.1f}\n")
                f.write(f"Pointer Altura: {h_ptr:.1f}\n")
                f.write(f"Diferencia (HR - Ptr): {diff:.1f}\n")
                f.write(f"\nRectángulos (width, height):\n")
                for i, (w, h) in enumerate(rects, 1):
                    f.write(f"{i}. ({w}, {h})\n")
            
            # Crear figura comparativa
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # HR - placements son [(rect, pos), ...] donde rect=(w,h) y pos=(x,y)
            ax1.set_xlim(0, W)
            ax1.set_ylim(0, max(h_hr, h_ptr) + 5)
            ax1.set_aspect('equal')
            ax1.set_title(f"HR - Altura: {h_hr:.1f}", fontsize=14, fontweight='bold')
            for rect_idx, (rect, pos) in enumerate(plc_hr):
                w, h = rect
                x, y = pos
                rect_patch = mpatches.Rectangle((x, y), w, h, linewidth=1, 
                                               edgecolor='black', facecolor=plt.cm.tab20(rect_idx % 20))
                ax1.add_patch(rect_patch)
            ax1.axhline(y=h_hr, color='red', linestyle='--', linewidth=2, label=f'Altura={h_hr:.1f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pointer - placements son [(rect, pos), ...] donde rect=(w,h) y pos=(x,y)
            ax2.set_xlim(0, W)
            ax2.set_ylim(0, max(h_hr, h_ptr) + 5)
            ax2.set_aspect('equal')
            ax2.set_title(f"Pointer - Altura: {h_ptr:.1f} (Mejor por {diff:.1f})", 
                         fontsize=14, fontweight='bold', color='green')
            for rect_idx, (rect, pos) in enumerate(plc_ptr):
                w, h = rect
                x, y = pos
                rect_patch = mpatches.Rectangle((x, y), w, h, linewidth=1, 
                                               edgecolor='black', facecolor=plt.cm.tab20(rect_idx % 20))
                ax2.add_patch(rect_patch)
            ax2.axhline(y=h_ptr, color='green', linestyle='--', linewidth=2, label=f'Altura={h_ptr:.1f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f"Caso donde Pointer superó a HR - Categoría {categoria} Problema #{problema_idx}", 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        print(f"  → Guardados en: {mejores_dir}/")
    
    return resultados


# ============================================================
# GENERAR BOXPLOT POR CATEGORÍA
# ============================================================
def generar_boxplot_categoria(
    resultados: List[ResultadoAlgoritmo],
    categoria: str,
    save_path: Optional[str] = None,
    incluir_tabla: bool = False
):
    """Genera boxplot de distribución de alturas para una categoría."""
    # Convertir a DataFrame
    df = pd.DataFrame([
        {
            "Algoritmo": r.algoritmo,
            "Altura": r.altura,
            "Tiempo (ms)": r.tiempo_ms,
            "Memoria (MB)": r.memoria_mb
        }
        for r in resultados if r.categoria == categoria
    ])
    
    if df.empty:
        print(f"No hay datos para categoría {categoria}")
        return
    
    # Configurar estilo
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ordenar algoritmos: HR, Pointer, luego greedy
    orden = ["HR", "Pointer", "FFDH", "BFDH", "Next Fit", "Bottom-Left"]
    orden_filtrado = [alg for alg in orden if alg in df["Algoritmo"].unique()]
    
    # Colores por tipo de algoritmo
    colores = {
        "HR": "#2ecc71",           # Verde (baseline óptimo)
        "Pointer": "#3498db",      # Azul (ML)
        "FFDH": "#e74c3c",         # Rojo
        "BFDH": "#e67e22",         # Naranja
        "Next Fit": "#9b59b6",     # Púrpura
        "Bottom-Left": "#95a5a6"   # Gris
    }
    palette = [colores.get(alg, "#34495e") for alg in orden_filtrado]
    
    # Crear boxplot
    bp = sns.boxplot(
        data=df,
        x="Algoritmo",
        y="Altura",
        order=orden_filtrado,
        palette=palette,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6}
    )
    
    # Añadir línea de referencia (Href)
    Href = cat.CATEGORIES[categoria]["height"]
    ax.axhline(y=Href, color='red', linestyle='--', linewidth=2, label=f'Href={Href}')
    
    # Título y etiquetas
    ax.set_title(f"Distribución de Altura - Categoría {categoria}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Algoritmo", fontsize=12)
    ax.set_ylabel("Altura Final", fontsize=12)
    ax.legend(loc='upper right')
    
    # Añadir estadísticas como texto (opcional)
    if incluir_tabla:
        stats_text = "Resumen Estadístico\n" + "─" * 25 + "\n"
        for alg in orden_filtrado:
            datos_alg = df[df["Algoritmo"] == alg]["Altura"]
            tiempo_prom = df[df["Algoritmo"] == alg]["Tiempo (ms)"].mean()
            memoria_prom = df[df["Algoritmo"] == alg]["Memoria (MB)"].mean()
            gap = ((datos_alg.mean() - Href) / Href * 100)
            
            stats_text += f"{alg:12s} | {datos_alg.mean():5.1f} | "
            stats_text += f"{tiempo_prom:6.1f}ms | {memoria_prom:5.1f}MB | {gap:+.1f}%\n"
        
        # Añadir texto en la esquina
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace'
        )
    
    plt.tight_layout()
    
    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Boxplot guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================
# GENERAR TABLA ESTADÍSTICA
# ============================================================
def generar_tabla_estadistica(
    resultados: List[ResultadoAlgoritmo],
    categoria: str
) -> pd.DataFrame:
    """Genera tabla estadística con métricas clave."""
    # Filtrar por categoría
    datos_cat = [r for r in resultados if r.categoria == categoria]
    
    if not datos_cat:
        return pd.DataFrame()
    
    # Agrupar por algoritmo
    df = pd.DataFrame([vars(r) for r in datos_cat])
    
    # Calcular estadísticas
    Href = cat.CATEGORIES[categoria]["height"]
    
    # Obtener altura promedio de HR para calcular Gap vs HR
    altura_hr = None
    if "HR" in df["algoritmo"].unique():
        altura_hr = df[df["algoritmo"] == "HR"]["altura"].mean()
    
    tabla = []
    for algoritmo in df["algoritmo"].unique():
        datos_alg = df[df["algoritmo"] == algoritmo]
        
        alturas = datos_alg["altura"]
        tiempos = datos_alg["tiempo_ms"]
        memorias = datos_alg["memoria_mb"]
        
        gap_href = ((alturas.mean() - Href) / Href * 100)
        
        fila = {
            "Algoritmo": algoritmo,
            "Tiempo (ms)": f"{tiempos.mean():.2f} ± {tiempos.std():.2f}",
            "Memoria (MB)": f"{memorias.mean():.2f}",
            "Altura": f"{alturas.mean():.1f}",
            "Gap vs Href": f"{gap_href:+.1f}%"
        }
        
        # Agregar Gap vs HR si no es el algoritmo HR
        if altura_hr is not None and algoritmo != "HR":
            gap_hr = ((alturas.mean() - altura_hr) / altura_hr * 100)
            fila["Gap vs HR"] = f"{gap_hr:+.1f}%"
        
        tabla.append(fila)
    
    return pd.DataFrame(tabla)


# ============================================================
# ANÁLISIS MULTI-CATEGORÍA
# ============================================================
def analizar_todas_categorias(
    categorias: List[str],
    n_problemas_por_cat: int = 50,
    device: str = "cpu",
    output_dir: str = "img/complexity_analysis",
    incluir_tabla_en_boxplot: bool = False,
    guardar_mejores_casos: bool = True
):
    """Ejecuta análisis completo para múltiples categorías.
    
    Genera:
    - Boxplot por cada categoría
    - Tabla comparativa unificada
    - Gráficos de escalabilidad (tiempo vs N, memoria vs N)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ANÁLISIS MULTI-CATEGORÍA")
    print(f"{'='*80}")
    print(f"Categorías: {', '.join(categorias)}")
    print(f"Problemas por categoría: {n_problemas_por_cat}")
    print(f"Output: {output_dir}/")
    
    todos_resultados = []
    
    # Analizar cada categoría
    for categoria in categorias:
        # Buscar modelo para esta categoría
        model_path = None
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) 
                          if f.startswith(f"pointer_{categoria.lower()}") and f.endswith(".pth")]
            if model_files:
                model_files.sort(reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
        
        # Analizar
        resultados = analizar_categoria(
            categoria=categoria,
            n_problemas=n_problemas_por_cat,
            model_path=model_path,
            device=device,
            save_mejor_casos=guardar_mejores_casos,
            output_dir=output_dir
        )
        todos_resultados.extend(resultados)
        
        # Generar boxplot para esta categoría
        boxplot_path = os.path.join(output_dir, f"boxplot_{categoria}.png")
        generar_boxplot_categoria(resultados, categoria, boxplot_path, incluir_tabla=incluir_tabla_en_boxplot)
        
        # Generar tabla para esta categoría
        tabla = generar_tabla_estadistica(resultados, categoria)
        print(f"\n{tabla.to_string(index=False)}")
        
        # Guardar tabla como CSV
        tabla_path = os.path.join(output_dir, f"tabla_{categoria}.csv")
        tabla.to_csv(tabla_path, index=False)
        print(f"✓ Tabla guardada: {tabla_path}")
    
    # Generar tabla unificada
    print(f"\n{'='*80}")
    print("TABLA UNIFICADA")
    print(f"{'='*80}")
    generar_tabla_unificada(todos_resultados, output_dir)
    
    # Generar gráficos de escalabilidad
    print(f"\n{'='*80}")
    print("GRÁFICOS DE ESCALABILIDAD")
    print(f"{'='*80}")
    generar_graficos_escalabilidad(todos_resultados, output_dir)
    
    # Generar resumen global
    print(f"\n{'='*80}")
    print("RESUMEN GLOBAL")
    print(f"{'='*80}")
    generar_resumen_global(todos_resultados, output_dir)
    
    print(f"\n{'='*80}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*80}")
    print(f"Archivos generados en: {output_dir}/")


def generar_tabla_unificada(resultados: List[ResultadoAlgoritmo], output_dir: str):
    """Genera tabla con todas las categorías."""
    if not resultados:
        print("No hay resultados para generar tabla unificada")
        return
    
    df = pd.DataFrame([vars(r) for r in resultados])
    
    tabla_unificada = []
    
    for categoria in sorted(df["categoria"].unique()):
        Href = cat.CATEGORIES[categoria]["height"]
        
        # Obtener altura promedio de HR para esta categoría
        df_cat = df[df["categoria"] == categoria]
        altura_hr = None
        if "HR" in df_cat["algoritmo"].unique():
            altura_hr = df_cat[df_cat["algoritmo"] == "HR"]["altura"].mean()
        
        for algoritmo in df["algoritmo"].unique():
            datos = df[(df["categoria"] == categoria) & (df["algoritmo"] == algoritmo)]
            
            if datos.empty:
                continue
            
            alturas = datos["altura"]
            tiempos = datos["tiempo_ms"]
            memorias = datos["memoria_mb"]
            
            gap_href = ((alturas.mean() - Href) / Href * 100)
            
            fila = {
                "Categoría": categoria,
                "Algoritmo": algoritmo,
                "Tiempo (ms)": f"{tiempos.mean():.1f}",
                "Memoria (MB)": f"{memorias.mean():.1f}",
                "Altura": f"{alturas.mean():.1f}",
                "Gap vs Href": f"{gap_href:+.1f}%"
            }
            
            # Agregar Gap vs HR si no es el algoritmo HR
            if altura_hr is not None and algoritmo != "HR":
                gap_hr = ((alturas.mean() - altura_hr) / altura_hr * 100)
                fila["Gap vs HR"] = f"{gap_hr:+.1f}%"
            
            tabla_unificada.append(fila)
    
    tabla_df = pd.DataFrame(tabla_unificada)
    
    # Guardar
    tabla_path = os.path.join(output_dir, "tabla_unificada.csv")
    tabla_df.to_csv(tabla_path, index=False)
    print(f"✓ Tabla unificada guardada: {tabla_path}")
    
    # Imprimir por categoría
    for categoria in sorted(df["categoria"].unique()):
        print(f"\n--- Categoría {categoria} ---")
        subtabla = tabla_df[tabla_df["Categoría"] == categoria].drop(columns=["Categoría"])
        print(subtabla.to_string(index=False))


def generar_resumen_global(resultados: List[ResultadoAlgoritmo], output_dir: str):
    """Genera resumen global con todas las métricas agregadas."""
    if not resultados:
        print("No hay resultados para generar resumen global")
        return
    
    df = pd.DataFrame([vars(r) for r in resultados])
    
    print("\n📊 MÉTRICAS GLOBALES (todas las categorías)")
    print("─" * 80)
    
    # Métricas por algoritmo
    for algoritmo in ["HR", "Pointer", "FFDH", "BFDH", "Next Fit", "Bottom-Left"]:
        if algoritmo not in df["algoritmo"].unique():
            continue
        
        datos_alg = df[df["algoritmo"] == algoritmo]
        
        tiempo_prom = datos_alg["tiempo_ms"].mean()
        memoria_prom = datos_alg["memoria_mb"].mean()
        altura_prom = datos_alg["altura"].mean()
        
        print(f"\n{algoritmo}:")
        print(f"  • Tiempo promedio: {tiempo_prom:.2f} ms")
        print(f"  • Memoria promedio: {memoria_prom:.2f} MB")
        print(f"  • Altura promedio: {altura_prom:.2f}")
        
        # Gap vs Href promedio
        gaps_href = []
        for categoria in df["categoria"].unique():
            Href = cat.CATEGORIES[categoria]["height"]
            datos_cat = datos_alg[datos_alg["categoria"] == categoria]
            if not datos_cat.empty:
                gap = ((datos_cat["altura"].mean() - Href) / Href * 100)
                gaps_href.append(gap)
        if gaps_href:
            print(f"  • Gap vs Href promedio: {np.mean(gaps_href):+.2f}%")
        
        # Gap vs HR (solo para no-HR)
        if algoritmo != "HR" and "HR" in df["algoritmo"].unique():
            gaps_hr = []
            for categoria in df["categoria"].unique():
                df_cat = df[df["categoria"] == categoria]
                altura_hr = df_cat[df_cat["algoritmo"] == "HR"]["altura"].mean()
                altura_alg = df_cat[df_cat["algoritmo"] == algoritmo]["altura"].mean()
                if not np.isnan(altura_hr) and not np.isnan(altura_alg):
                    gap = ((altura_alg - altura_hr) / altura_hr * 100)
                    gaps_hr.append(gap)
            if gaps_hr:
                print(f"  • Gap vs HR promedio: {np.mean(gaps_hr):+.2f}%")
    
    # Casos de superioridad Pointer vs HR
    if "Pointer" in df["algoritmo"].unique() and "HR" in df["algoritmo"].unique():
        print(f"\n🎯 CASOS DE SUPERIORIDAD (Pointer vs HR):")
        casos_totales = 0
        casos_mejor = 0
        casos_empate = 0
        
        for categoria in df["categoria"].unique():
            df_cat = df[df["categoria"] == categoria]
            df_hr = df_cat[df_cat["algoritmo"] == "HR"]
            df_ptr = df_cat[df_cat["algoritmo"] == "Pointer"]
            
            # Comparar problema por problema
            for idx in df_hr["problema_idx"].unique():
                hr_altura = df_hr[df_hr["problema_idx"] == idx]["altura"].values
                ptr_altura = df_ptr[df_ptr["problema_idx"] == idx]["altura"].values
                
                if len(hr_altura) > 0 and len(ptr_altura) > 0:
                    casos_totales += 1
                    if ptr_altura[0] < hr_altura[0]:
                        casos_mejor += 1
                    elif abs(ptr_altura[0] - hr_altura[0]) < 0.01:  # Empate (tolerancia)
                        casos_empate += 1
        
        if casos_totales > 0:
            casos_peor = casos_totales - casos_mejor - casos_empate
            porcentaje_mejor = (casos_mejor / casos_totales) * 100
            porcentaje_empate = (casos_empate / casos_totales) * 100
            porcentaje_peor = (casos_peor / casos_totales) * 100
            
            print(f"  • Pointer mejor que HR: {casos_mejor}/{casos_totales} casos ({porcentaje_mejor:.1f}%)")
            print(f"  • Pointer igual a HR: {casos_empate}/{casos_totales} casos ({porcentaje_empate:.1f}%)")
            print(f"  • HR mejor que Pointer: {casos_peor}/{casos_totales} casos ({porcentaje_peor:.1f}%)")
    
    print("\n" + "─" * 80)


def generar_graficos_escalabilidad(resultados: List[ResultadoAlgoritmo], output_dir: str):
    """Genera gráficos de tiempo y memoria vs número de rectángulos."""
    df = pd.DataFrame([vars(r) for r in resultados])
    
    # Agrupar por algoritmo y número de rectángulos
    grouped = df.groupby(["algoritmo", "num_rectangulos"]).agg({
        "tiempo_ms": ["mean", "std"],
        "memoria_mb": ["mean", "std"]
    }).reset_index()
    
    # Gráfico 1: Tiempo vs N
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algoritmo in grouped["algoritmo"].unique():
        datos_alg = grouped[grouped["algoritmo"] == algoritmo]
        ax.plot(
            datos_alg["num_rectangulos"],
            datos_alg["tiempo_ms"]["mean"],
            marker='o',
            label=algoritmo,
            linewidth=2
        )
        ax.fill_between(
            datos_alg["num_rectangulos"],
            datos_alg["tiempo_ms"]["mean"] - datos_alg["tiempo_ms"]["std"],
            datos_alg["tiempo_ms"]["mean"] + datos_alg["tiempo_ms"]["std"],
            alpha=0.2
        )
    
    ax.set_xlabel("Número de Rectángulos (N)", fontsize=12)
    ax.set_ylabel("Tiempo de Ejecución (ms)", fontsize=12)
    ax.set_title("Escalabilidad Temporal: Tiempo vs Tamaño del Problema", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    tiempo_path = os.path.join(output_dir, "escalabilidad_tiempo.png")
    plt.savefig(tiempo_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de tiempo guardado: {tiempo_path}")
    plt.close()
    
    # Gráfico 2: Memoria vs N
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algoritmo in grouped["algoritmo"].unique():
        datos_alg = grouped[grouped["algoritmo"] == algoritmo]
        ax.plot(
            datos_alg["num_rectangulos"],
            datos_alg["memoria_mb"]["mean"],
            marker='s',
            label=algoritmo,
            linewidth=2
        )
    
    ax.set_xlabel("Número de Rectángulos (N)", fontsize=12)
    ax.set_ylabel("Uso de Memoria (MB)", fontsize=12)
    ax.set_title("Escalabilidad Espacial: Memoria vs Tamaño del Problema", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    memoria_path = os.path.join(output_dir, "escalabilidad_memoria.png")
    plt.savefig(memoria_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de memoria guardado: {memoria_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    """
    Uso:
        python analisis_complejidad.py <cat1> <cat2> ... [n_problemas]
        
    Ejemplos:
        python analisis_complejidad.py C1 C2 C3 50
        python analisis_complejidad.py C1 100
    """
    # Parsear argumentos
    categorias = []
    n_problemas = 50  # Default
    
    for arg in sys.argv[1:]:
        arg_upper = arg.upper()
        if arg_upper in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            categorias.append(arg_upper)
        else:
            try:
                n_problemas = int(arg)
            except ValueError:
                print(f"Argumento ignorado: {arg}")
    
    if not categorias:
        print("Uso: python analisis_complejidad.py <cat1> <cat2> ... [n_problemas]")
        print("Ejemplo: python analisis_complejidad.py C1 C2 C3 50")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    analizar_todas_categorias(
        categorias=categorias,
        n_problemas_por_cat=n_problemas,
        device=device
    )
