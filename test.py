import os
# Quick local workaround for Intel OpenMP duplicate runtime on Windows.
# NOTE: Prefer fixing your environment (see assistant message) instead of relying on this.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import categories as cat
import generator as gen
import hr_algorithm as hr
from pointer_model import SPPPointerModel
from hr_pointer import heuristic_pointer_wrapper

categoria = "C1"

# Cargar modelo entrenado
model = SPPPointerModel()
checkpoint = torch.load("models/pointer_C1.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

# Generar un problema de prueba
problemas, W, H = gen.generate_problems_guillotine(categoria, 1, export=False)
rects = problemas[0]

placements, altura, seq, all_states, Y_rect, _, _ = heuristic_pointer_wrapper(
    rects,
    container_width=W,
    model=model,
    category=categoria,
    device="cpu",
)

print("Altura:", altura)
print("Secuencia dinámica (seq):", seq)
print("Rectángulos en orden colocado:", [r for (r,pos) in placements])

# Dibujar
Href = cat.CATEGORIES[categoria]["height"]
hr.visualizar_packing(placements, container_width=W, container_height=Href, show=True)