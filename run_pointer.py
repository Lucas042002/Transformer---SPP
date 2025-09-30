"""Pipeline rápido para entrenar y probar el modelo pointer con problemas generados.

Uso (ejemplo):
  python run_pointer.py --categoria C1 --num_problemas 80 --epochs 15

Genera problemas con el generador existente, construye dataloaders pointer,
entrena el modelo y hace una inferencia de ejemplo sobre un nuevo problema.
"""
from __future__ import annotations

import argparse
import torch
import hr_algorithm as hr

import categories as cat
import generator as gen  # Asumiendo ya existe en tu repo
from pointer_model import SPPPointerModel
from pointer_training import build_pointer_dataloaders, train_pointer_model
import hr_pointer as hp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categoria", default="C1")
    parser.add_argument("--num_problemas", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--teacher", choices=["fillratio", "hr", "hr_algo"], default="hr", help="Maestro para generar trayectorias (incluye 'hr_algo' para usar hr_algorithm.hr_packing)")
    parser.add_argument("--visualizar", action="store_true", help="Si se pasa, muestra la visualización del packing del problema de prueba")
    parser.add_argument("--num_visualizaciones", type=int, default=1, help="Cantidad de problemas de prueba a visualizar (solo si --visualizar)")
    args = parser.parse_args()

    categoria = args.categoria
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"Generando {args.num_problemas} problemas de {categoria}...")
    problemas, W, H = gen.generate_problems_guillotine(categoria, args.num_problemas, export=False)
    print("Ejemplo problema 0:", problemas[0][:5])

    print("Construyendo dataloaders pointer...")
    train_loader, val_loader, ntr, nv = build_pointer_dataloaders(
        problemas,
        categoria,
        batch_size=args.batch_size,
        teacher=args.teacher,
    )
    print(f"Pasos train: {ntr}  val: {nv}")

    model = SPPPointerModel()
    history = train_pointer_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Guardar modelo simple
    torch.save({"state_dict": model.state_dict(), "categoria": categoria}, f"models/pointer_{categoria}.pth")
    print(f"Modelo pointer guardado en models/pointer_{categoria}.pth")

    # Inferencia demo en un nuevo problema
    problemas_test, Wt, Ht = gen.generate_problems_guillotine(categoria, max(1, args.num_visualizaciones), export=False)
    for i in range(min(args.num_visualizaciones, len(problemas_test))):
        test_rects = problemas_test[i]
        placements, altura, seq, all_states, Y_rect, _, _ = hp.heuristic_pointer_wrapper(
            test_rects,
            container_width=Wt,
            model=model,
            category=categoria,
            device=device,
        )
        print(f"\n=== Problema de prueba {i+1} ===")
        print("Rectángulos originales:", test_rects)
        print("Altura resultante pointer:", altura)
        # Secuencia real de rectángulos colocados (en orden) tomando placements
        rects_en_orden = [r for (r, pos) in placements]
        print("Rectángulos en orden colocado:", rects_en_orden)
        print("Indices dinámicos elegidos (seq):", seq)
        if args.visualizar:
            # Usar altura de referencia de la categoría para dibujar línea roja
            Href = cat.CATEGORIES[categoria]["height"]
            hr.visualizar_packing(placements, container_width=Wt, container_height=Href, show=True)

    


if __name__ == "__main__":
    main()
