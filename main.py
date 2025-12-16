import hr_algorithm as hr
from pointer_model import SPPPointerModel
from pointer_training import build_pointer_dataloaders, train_pointer_model
from test_pointer import (
    test_modelo_pointer,
    test_y_visualizar_problema,
    visualizar_mejores_peores_casos,
    test_comparacion_completa
)
import generator as gen
import torch
import categories as cat
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




# --------------------------------------------------
# Script de entrenamiento y testing principal
# --------------------------------------------------
if __name__ == "__main__":
    """Script principal: entrenar, testear o visualizar el modelo pointer.
    
    Uso:
        python main.py <categoria>                          # Entrenar modelo (C1-C7)
        python main.py <categoria> test                     # Testear último modelo entrenado
        python main.py <categoria> test <path> <n>          # Testear modelo específico con n problemas
        python main.py <categoria> compare                  # Comparar con HR y greedy
        python main.py <categoria> compare <path> <n>       # Comparar modelo específico con n problemas
        python main.py <categoria> visual                   # Visualizar un problema aleatorio
        python main.py <categoria> visual <path>            # Visualizar con modelo específico
        python main.py <categoria> best-worst               # Mostrar mejores y peores casos
        python main.py <categoria> best-worst <path> <n>    # Con modelo y n problemas específicos
        python main.py complexity <cat1> <cat2> ... <n>     # Análisis de complejidad multi-categoría
    
    Ejemplos:
        python main.py C1                    # Entrenar modelo para C1
        python main.py C4 test               # Testear modelo C4
        python main.py C2 compare            # Comparar modelo C2
        python main.py complexity C1 C2 C3 C4 500  # Análisis de complejidad
    """
    
    # Detectar categoría (primer argumento)
    CATEGORIA = "C1"  # Default
    modo = "train"  # Por defecto
    modelo_path = None
    n_problemas = None
    
    if len(sys.argv) > 1:
        # Primer argumento: puede ser categoría o modo
        arg1 = sys.argv[1].upper()
        
        # MODO ESPECIAL: Análisis de complejidad multi-categoría
        if arg1 == "COMPLEXITY":
            import analisis_complejidad as ac
            
            categorias = []
            n_probs = 50  # Default
            
            for arg in sys.argv[2:]:
                arg_upper = arg.upper()
                if arg_upper in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
                    categorias.append(arg_upper)
                else:
                    try:
                        n_probs = int(arg)
                    except ValueError:
                        print(f"Argumento ignorado: {arg}")
            
            if not categorias:
                print("Uso: python main.py complexity <cat1> <cat2> ... [n_problemas]")
                print("Ejemplo: python main.py complexity C1 C2 C3 50")
                sys.exit(1)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            ac.analizar_todas_categorias(
                categorias=categorias,
                n_problemas_por_cat=n_probs,
                device=device
            )
            sys.exit(0)
        
        if arg1 in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            CATEGORIA = arg1
            # Si hay más argumentos, el siguiente es el modo
            if len(sys.argv) > 2:
                modo = sys.argv[2].lower()
                if len(sys.argv) > 3:
                    modelo_path = sys.argv[3]
                if len(sys.argv) > 4:
                    try:
                        n_problemas = int(sys.argv[4])
                    except ValueError:
                        print(f"Advertencia: '{sys.argv[4]}' no es un número válido, usando default")
        else:
            # Si no es categoría, es modo (backward compatibility)
            modo = arg1.lower()
            if len(sys.argv) > 2:
                modelo_path = sys.argv[2]
            if len(sys.argv) > 3:
                try:
                    n_problemas = int(sys.argv[3])
                except ValueError:
                    print(f"Advertencia: '{sys.argv[3]}' no es un número válido, usando default")
    
    # ============================================================
    # MODO TESTING
    # ============================================================
    if modo == "test":
        print("\n" + "="*80)
        print("TESTING DEL MODELO POINTER")
        print("="*80)
        
        N_PROBLEMAS_TEST = n_problemas if n_problemas else 20
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\nConfiguración de testing:")
        print(f"  Categoría: {CATEGORIA}")
        print(f"  Problemas de test: {N_PROBLEMAS_TEST}")
        print(f"  Device: {DEVICE}")
        
        if modelo_path:
            print(f"  Modelo: {modelo_path}")
        else:
            print(f"  Modelo: Auto-detectar último modelo")
        
        # Ejecutar testing
        resultados = test_modelo_pointer(
            model_path=modelo_path,
            n_problemas=N_PROBLEMAS_TEST,
            categoria=CATEGORIA,
            device=DEVICE
        )
        
        print(f"\nTesting completado!")
        sys.exit(0)
    
    # ============================================================
    # MODO COMPARACIÓN COMPLETA (Pointer vs HR vs Greedy)
    # ============================================================
    elif modo == "compare":
        print("\n" + "="*80)
        print("COMPARACIÓN COMPLETA: POINTER vs HR vs GREEDY")
        print("="*80)
        
        N_PROBLEMAS_TEST = n_problemas if n_problemas else 20
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\nConfiguración:")
        print(f"  Categoría: {CATEGORIA}")
        print(f"  Problemas: {N_PROBLEMAS_TEST}")
        print(f"  Device: {DEVICE}")
        print(f"  Algoritmos: HR, Pointer, FFDH, BFDH, Next Fit, Bottom-Left")
        
        if modelo_path:
            print(f"  Modelo: {modelo_path}")
        else:
            print(f"  Modelo: Auto-detectar último modelo")
        
        # Ejecutar comparación completa
        stats = test_comparacion_completa(
            model_path=modelo_path,
            n_problemas=N_PROBLEMAS_TEST,
            categoria=CATEGORIA,
            device=DEVICE,
            incluir_greedy=True
        )
        
        print(f"\nComparación completada!")
        sys.exit(0)
    
    # ============================================================
    # MODO VISUALIZACIÓN
    # ============================================================
    elif modo == "visual":
        print("\n" + "="*80)
        print("VISUALIZACIÓN DE PROBLEMA INDIVIDUAL")
        print("="*80)
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detectar modelo si no se especifica
        if modelo_path is None:
            models_dir = "models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) 
                              if f.startswith(f"pointer_{CATEGORIA.lower()}") and f.endswith(".pth")]
                if model_files:
                    model_files.sort(reverse=True)
                    modelo_path = os.path.join(models_dir, model_files[0])
                    print(f"Auto-detectado modelo: {modelo_path}")
                else:
                    print(f"No se encontraron modelos en {models_dir}/ para categoría {CATEGORIA}")
                    sys.exit(1)
            else:
                print(f"Directorio {models_dir}/ no existe")
                sys.exit(1)
        
        print(f"\nConfiguración:")
        print(f"  Categoría: {CATEGORIA}")
        print(f"  Device: {DEVICE}")
        print(f"  Modelo: {modelo_path}")
        
        # Visualizar problema aleatorio con comparación
        test_y_visualizar_problema(
            model_path=modelo_path,
            problema_rects=None,  # Genera uno aleatorio
            categoria=CATEGORIA,
            device=DEVICE,
            comparar_hr=True  # Mostrar comparación con HR
        )
        
        print(f"\nVisualización completada!")
        sys.exit(0)
    
    # ============================================================
    # MODO MEJORES/PEORES CASOS
    # ============================================================
    elif modo == "best-worst":
        print("\n" + "="*80)
        print("ANÁLISIS DE MEJORES Y PEORES CASOS")
        print("="*80)
        
        N_PROBLEMAS = n_problemas if n_problemas else 20
        N_MOSTRAR = 3  # Cuántos mejores/peores mostrar
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detectar modelo si no se especifica
        if modelo_path is None:
            models_dir = "models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) 
                              if f.startswith(f"pointer_{CATEGORIA.lower()}") and f.endswith(".pth")]
                if model_files:
                    model_files.sort(reverse=True)
                    modelo_path = os.path.join(models_dir, model_files[0])
                    print(f"Auto-detectado modelo: {modelo_path}")
                else:
                    print(f"No se encontraron modelos en {models_dir}/ para categoría {CATEGORIA}")
                    sys.exit(1)
            else:
                print(f"Directorio {models_dir}/ no existe")
                sys.exit(1)
        
        print(f"\nConfiguración:")
        print(f"  Categoría: {CATEGORIA}")
        print(f"  Problemas a analizar: {N_PROBLEMAS}")
        print(f"  Casos a mostrar: {N_MOSTRAR}")
        print(f"  Device: {DEVICE}")
        print(f"  Modelo: {modelo_path}")
        
        # Ejecutar análisis
        visualizar_mejores_peores_casos(
            model_path=modelo_path,
            n_problemas=N_PROBLEMAS,
            n_mostrar=N_MOSTRAR,
            categoria=CATEGORIA,
            device=DEVICE
        )
        
        print(f"\nAnálisis completado!")
        sys.exit(0)
    
    # ============================================================
    # MODO ENTRENAMIENTO (por defecto)
    # ============================================================
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO DEL MODELO POINTER")
    print("="*80)
    
    # Configuración
    NUM_PROBLEMAS = 500 
    BATCH_SIZE = 16
    EPOCHS = 200
    
    # HYPERPARÁMETROS DEL MODELO
    NUM_ENC_LAYERS = 4
    NUM_HEADS = 8
    DROP_RATE = 0.1
    
    # HYPERPARÁMETROS DE ENTRENAMIENTO
    LEARNING_RATE = 1e-4   # 0.0001 (actual, conservador)
    # LEARNING_RATE = 3e-4   # 0.0003 (un poco más rápido)
    # LEARNING_RATE = 1e-3   # 0.001  (más agresivo)
    # LEARNING_RATE = 5e-5   # 0.00005 (muy conservador)

    WEIGHT_DECAY = 1e-4   # Regularización L2
    GRAD_CLIP = 1.0       # Gradient clipping
    EARLY_STOPPING_PATIENCE = 40  # Más paciencia con más datos (era 15)
    USE_WEIGHTED_LOSS = True  # Dar más importancia a pasos difíciles

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n CONFIGURACIÓN:")
    print(f"  Categoría: {CATEGORIA}")
    print(f"  Problemas de entrenamiento: {NUM_PROBLEMAS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Encoder layers: {NUM_ENC_LAYERS}")
    print(f"  Attention heads: {NUM_HEADS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Weighted loss: {USE_WEIGHTED_LOSS}")
    print(f"  Device: {DEVICE}")
    
    # Generar problemas de entrenamiento
    print(f"\n Generando {NUM_PROBLEMAS} problemas...")
    problems, ancho, alto = gen.generate_problems_guillotine(CATEGORIA, NUM_PROBLEMAS, export=False)
    print(f" {len(problems)} problemas generados")
    # print(f"  DEBUG - Tipo de problems: {type(problems)}")
    # print(f"  DEBUG - Primer problema: {problems[0][:3] if problems else 'N/A'}...")
    # print(f"  DEBUG - Tipo del primer rect: {type(problems[0][0]) if problems and problems[0] else 'N/A'}")
    
    # Crear dataloaders
    train_loader, val_loader, num_train, num_val = build_pointer_dataloaders(
        problems=problems,
        category=CATEGORIA,
        batch_size=BATCH_SIZE,
        val_split=0.1,
        shuffle=True,
        seed=42,
        augment_permutations=2  # Data Augmentation: 500 problemas → 1500 variantes (3x)
    )
    
    # Crear modelo
    print(f"\nCreando modelo...")
    model = SPPPointerModel(
        d_model=256,
        rect_feat_dim=10,
        space_feat_dim=19,  # Actualizado: 10 básicas + 1 fragmentation + 8 compatibilidad (incluye minimización altura)
        num_enc_layers=NUM_ENC_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=512,
        dropout=DROP_RATE,
        max_steps=cat.CATEGORIES[CATEGORIA]["num_items"]
    )
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Entrenar
    print(f"\n Iniciando entrenamiento...")
    history = train_pointer_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        grad_clip=GRAD_CLIP,
        categoria=CATEGORIA,
        guardar_modelo=True,
        num_enc_layers=NUM_ENC_LAYERS,
        num_heads=NUM_HEADS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        use_weighted_loss=USE_WEIGHTED_LOSS  # Activar weighted loss
    )
    
    # Mostrar resultados finales
    print(f"\nRESULTADOS FINALES:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Acc: {history['train_acc'][-1]:.3f}")
    print(f"  Val Acc: {history['val_acc'][-1]:.3f}")
    print(f"  Best Val Acc: {max(history['val_acc']):.3f}")
    print(f"\n{'='*80}")
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)

