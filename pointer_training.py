"""Entrenamiento del modelo pointer (SPPPointerModel) mediante Imitation Learning.

Proporciona:
 - Generación de trayectorias heurísticas (sin depender de hr_algorithm incompleto)
 - Dataset y collate para candidatos de longitud variable
 - Loop de entrenamiento con máscara de factibilidad

Heurística usada para generar targets (puedes reemplazar luego por HR real):
  * En cada paso se selecciona el rectángulo factible que maximiza area/(S_w * S_h) (fill ratio)

Pasos futuros posibles:
  * Reemplazar selección heurística por trayectorias de HR / mejor solución conocida
  * Añadir acción de rotación explícita (duplicar candidatos) 
  * Policy gradient / fine-tuning RL sobre altura final
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import categories as cat
import states as st
import hr_algorithm as hr  # <- para usar hr_packing como maestro real
from pointer_model import SPPPointerModel

Rect = Tuple[int, int]
Space = Tuple[int, int, int, int]


# --------------------------------------------------
# Utilidades geométricas básicas
# --------------------------------------------------
def rect_fits_in_space(rect: Rect, space: Space):
    rw, rh = rect
    _, _, w, h = space
    if rw <= w and rh <= h:
        return True, 0
    if rh <= w and rw <= h:
        return True, 1
    return False, -1


def place_rect(space: Space, rect: Rect):
    fits, rot = rect_fits_in_space(rect, space)
    if not fits:
        return False, (-1, -1), rect, rot
    rw, rh = rect
    if rot == 1:
        rw, rh = rh, rw
    x, y, w, h = space
    return True, (x, y), (rw, rh), rot


def divide_space(space: Space, placed_size: Rect, pos):
    x, y, w, h = space
    rw, rh = placed_size
    rx, ry = pos
    S1 = (x, ry + rh, w, (y + h) - (ry + rh))
    S2 = (rx + rw, y, (x + w) - (rx + rw), rh)
    return S1, S2


def limpiar_spaces(spaces: List[Space]):
    return [s for s in spaces if s[2] > 0 and s[3] > 0]


# --------------------------------------------------
# Generación de trayectorias (maestros: fillratio | hr)
# --------------------------------------------------
@dataclass
class PointerStep:
    rect_feats: torch.Tensor  # (N, F)
    rect_mask: torch.Tensor   # (N,) bool (factibles)
    space_feat: torch.Tensor  # (F_space,)
    target: int               # índice elegido (0..N-1) solo entre factibles
    step_idx: int


def build_pointer_trajectory_fillratio(rects: List[Rect], container_width: int, category: str) -> List[PointerStep]:
    Href = cat.CATEGORIES[category]["height"]
    max_height_virtual = Href * 5
    spaces: List[Space] = [(0, 0, container_width, max_height_virtual)]
    remaining = rects.copy()
    steps: List[PointerStep] = []
    step = 0

    while remaining:
        active_space = spaces[-1]
        # Features de espacio
        space_feat_vec = st._space_features_sin_seqid(active_space, container_width, Href, include_xy=True, S_active=active_space)

        rect_feats_list = []
        feasible_mask = []
        fill_ratios = []
        Sw, Sh = active_space[2], active_space[3]
        S_area = Sw * Sh if Sw > 0 and Sh > 0 else 1.0

        for r in remaining:
            feats = st._rect_features_sin_seqid(r, active_space, container_width, Href)
            rect_feats_list.append(feats)
            feasible = (feats[5] > 0.5) or (feats[6] > 0.5)
            feasible_mask.append(1 if feasible else 0)
            if feasible:
                area = r[0]*r[1]
                fill_ratios.append(area / S_area)
            else:
                fill_ratios.append(-1.0)

        if not any(feasible_mask):
            # descartar espacio y continuar
            spaces.pop()
            if not spaces:
                break
            continue

        # Selección heurística: mayor fill ratio
        target_idx = max(range(len(remaining)), key=lambda i: fill_ratios[i])

        step_obj = PointerStep(
            rect_feats=torch.tensor(rect_feats_list, dtype=torch.float32),
            rect_mask=torch.tensor(feasible_mask, dtype=torch.bool),
            space_feat=torch.tensor(space_feat_vec, dtype=torch.float32),
            target=target_idx,
            step_idx=step,
        )
        steps.append(step_obj)

        # Efectuar colocación
        chosen_rect = remaining[target_idx]
        ok, pos, placed_size, rot = place_rect(active_space, chosen_rect)
        if ok:
            S1, S2 = divide_space(active_space, placed_size, pos)
            spaces.pop()
            spaces.extend([S1, S2])
            spaces = limpiar_spaces(spaces)
            remaining.pop(target_idx)
        else:
            # seguridad: si falla remove rect para no ciclo infinito
            remaining.pop(target_idx)
        step += 1

    return steps


def build_pointer_trajectory_hr(rects: List[Rect], container_width: int, category: str) -> List[PointerStep]:
    """Genera trayectoria usando una variante simple del HR como maestro:
    - Ordena rectángulos restantes por área (desc) cada paso (estrategia greedy)
    - Recorre espacios actuales en orden de aparición para encontrar el primero que acepta algún rect
    - El rect elegido es el primero que cabe (sin volver a rotar lista base)
    El target se refiere al índice del rectángulo elegido dentro de la LISTA ORIGINAL remaining (no reordenada) para que coincida con rect_feats.
    """
    Href = cat.CATEGORIES[category]["height"]
    max_height_virtual = Href * 5
    spaces: List[Space] = [(0, 0, container_width, max_height_virtual)]
    remaining = rects.copy()
    steps: List[PointerStep] = []
    step = 0

    while remaining:
        # Elegir un espacio que permita colocar al menos un rectángulo (siguiendo orden)
        active_space = None
        for sp in spaces:
            # comprobar factibilidad rápida
            if any(rect_fits_in_space(r, sp)[0] for r in remaining):
                active_space = sp
                break
        if active_space is None:
            # No cabe nada en ningún espacio -> terminar
            break

        # Lista auxiliar ordenada por área para decidir maestro
        ordered = sorted(enumerate(remaining), key=lambda x: x[1][0] * x[1][1], reverse=True)
        chosen_orig_index = None
        for original_idx, r in ordered:
            fits, _ = rect_fits_in_space(r, active_space)
            if fits:
                chosen_orig_index = original_idx
                break
        if chosen_orig_index is None:
            # Ninguno cabe realmente (inconsistencia) -> eliminar espacio y continuar
            spaces.remove(active_space)
            if not spaces:
                break
            continue

        # Construir features RELATIVOS al active_space usando el orden actual de remaining
        space_feat_vec = st._space_features_sin_seqid(active_space, container_width, Href, include_xy=True, S_active=active_space)
        rect_feats_list = []
        feasible_mask = []
        for r in remaining:
            feats = st._rect_features_sin_seqid(r, active_space, container_width, Href)
            rect_feats_list.append(feats)
            feasible = (feats[5] > 0.5) or (feats[6] > 0.5)
            feasible_mask.append(1 if feasible else 0)

        step_obj = PointerStep(
            rect_feats=torch.tensor(rect_feats_list, dtype=torch.float32),
            rect_mask=torch.tensor(feasible_mask, dtype=torch.bool),
            space_feat=torch.tensor(space_feat_vec, dtype=torch.float32),
            target=chosen_orig_index,
            step_idx=step,
        )
        steps.append(step_obj)

        # Colocar el rect (usar chosen_orig_index) y dividir espacio
        chosen_rect = remaining[chosen_orig_index]
        ok, pos, placed_size, rot = place_rect(active_space, chosen_rect)
        if ok:
            S1, S2 = divide_space(active_space, placed_size, pos)
            # Sustituir espacio usado por sus divisiones (similar a estrategia simple)
            sp_idx = spaces.index(active_space)
            spaces.pop(sp_idx)
            spaces.extend([S1, S2])
            spaces = limpiar_spaces(spaces)
            remaining.pop(chosen_orig_index)
        else:
            # Falla inesperada: remover rect para no ciclar
            remaining.pop(chosen_orig_index)
        step += 1

    return steps


def build_pointer_trajectory_from_hr_algorithm(rects: List[Rect], container_width: int, category: str) -> List[PointerStep]:
    """Construye una trayectoria de PointerSteps usando directamente el algoritmo HR actual
    (función hr.hr_packing) que ya genera estados (X) y decisiones (Y).

    Mapping:
      - Cada "estado" producido por hr_packing es una lista [S_in, R_in]
          * S_in: lista de subespacios (cada vector con 12 feats, índice 10 = a_utilizar)
          * R_in: lista de rect features (12 feats, indices 5=fits, 6=fits_rot)
      - Y_rect es 1-based (1..N); se convierte a 0-based para el target del pointer.

    Estrategia:
      - Identificamos el subespacio activo buscando el vector con a_utilizar==1.
        Si no se encuentra marcaje (caso raro), usamos el primero.
      - Usamos TODA la lista de rectángulos del estado como candidatos en el mismo orden.
      - La máscara de factibilidad se deriva de (fits or fits_rot).
      - Si por alguna inconsistencia el target apunta a un índice fuera de rango o no factible,
        ese paso se descarta para evitar ruido.
    """
    steps: List[PointerStep] = []
    # Ejecutar HR para obtener estados y decisiones
    # Inicializamos espacios como en hr_packing: altura grande virtual (se hace dentro de hr_packing)
    rects_copy = rects.copy()
    spaces_init = [(0, 0, container_width, 1000)]  # 1000 se usa también en hr_algorithm
    placed, estados, Y_rect = hr.hr_packing(spaces=spaces_init, rects=rects_copy, category=category)

    if not estados or not Y_rect:
        return steps

    # Recorremos pasos. Y_rect puede tener longitud igual al número de colocaciones efectivas.
    # Nos aseguramos de no exceder.
    num_steps = min(len(estados), len(Y_rect))
    for step_idx in range(num_steps):
        state = estados[step_idx]
        target_1b = Y_rect[step_idx]  # 1-based
        if target_1b <= 0:
            continue  # ignorar decisiones inválidas
        target = target_1b - 1

        if not isinstance(state, (list, tuple)) or len(state) < 2:
            continue
        S_in, R_in = state[0], state[1]
        if not R_in:
            continue

        # Encontrar subespacio activo (a_utilizar==1 en índice 10)
        active_space_vec = None
        for sv in S_in:
            if len(sv) >= 11 and int(round(sv[10])) == 1:
                active_space_vec = sv
                break
        if active_space_vec is None:
            active_space_vec = S_in[0]

        # Construir máscara de factibilidad a partir de fits / fits_rot (indices 5 y 6)
        feasible_mask_list: List[int] = []
        for rv in R_in:
            if len(rv) < 7:
                feasible_mask_list.append(0)
            else:
                feasible = (rv[5] > 0.5) or (rv[6] > 0.5)
                feasible_mask_list.append(1 if feasible else 0)

        # Validar target
        if target >= len(R_in):
            continue
        if feasible_mask_list[target] == 0:
            # El HR eligió algo marcado como no factible (inconsistencia), descartamos
            continue

        # Convertir a tensores
        rect_feats_tensor = torch.tensor(R_in, dtype=torch.float32)
        rect_mask_tensor = torch.tensor(feasible_mask_list, dtype=torch.bool)
        space_feat_tensor = torch.tensor(active_space_vec, dtype=torch.float32)

        steps.append(PointerStep(
            rect_feats=rect_feats_tensor,
            rect_mask=rect_mask_tensor,
            space_feat=space_feat_tensor,
            target=target,
            step_idx=step_idx,
        ))

    return steps


def build_dataset_from_problems(
    problems: List[List[Rect]],
    category: str,
    teacher: Literal["fillratio", "hr", "hr_algo"] = "fillratio",
) -> List[PointerStep]:
    """Concatena trayectorias en una lista (plano) según maestro seleccionado.

    teacher opciones:
      - "fillratio": heurística de razón área/espacio
      - "hr": variante simple HR implementada localmente (orden por área + primer ajuste)
      - "hr_algo": usa directamente hr_algorithm.hr_packing para generar (estados, Y)
    """
    out: List[PointerStep] = []
    W = cat.CATEGORIES[category]["width"]
    for rects in problems:
        if teacher == "fillratio":
            out.extend(build_pointer_trajectory_fillratio(rects, W, category))
        elif teacher == "hr":
            out.extend(build_pointer_trajectory_hr(rects, W, category))
        elif teacher == "hr_algo":
            out.extend(build_pointer_trajectory_from_hr_algorithm(rects, W, category))
        else:
            raise ValueError(f"Teacher desconocido: {teacher}")
    return out


# --------------------------------------------------
# Dataset y collate
# --------------------------------------------------
class PointerStepsDataset(Dataset):
    def __init__(self, steps: List[PointerStep]):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        s = self.steps[idx]
        return {
            "rect_feats": s.rect_feats,  # (N,F)
            "rect_mask": s.rect_mask,    # (N,)
            "space_feat": s.space_feat,  # (F_space,)
            "target": s.target,          # int
            "step_idx": s.step_idx,      # int
        }


def pointer_collate(batch: List[Dict[str, torch.Tensor]]):
    # Determinar tamaños
    max_N = max(item["rect_feats"].shape[0] for item in batch)
    F = batch[0]["rect_feats"].shape[1]
    device = batch[0]["rect_feats"].device if batch[0]["rect_feats"].is_cuda else torch.device("cpu")

    rect_feats_batch = torch.zeros(len(batch), max_N, F, dtype=torch.float32, device=device)
    rect_mask_batch = torch.zeros(len(batch), max_N, dtype=torch.bool, device=device)
    space_feat_batch = torch.stack([item["space_feat"] for item in batch])  # (B,F_space)
    step_idx_batch = torch.tensor([item["step_idx"] for item in batch], dtype=torch.long, device=device)
    targets = torch.tensor([item["target"] for item in batch], dtype=torch.long, device=device)

    for b, item in enumerate(batch):
        N = item["rect_feats"].shape[0]
        rect_feats_batch[b, :N] = item["rect_feats"]
        rect_mask_batch[b, :N] = item["rect_mask"]

    return {
        "rect_feats": rect_feats_batch,
        "rect_mask": rect_mask_batch,
        "space_feat": space_feat_batch,
        "targets": targets,
        "step_idx": step_idx_batch,
    }


# --------------------------------------------------
# Entrenamiento
# --------------------------------------------------
def train_pointer_model(
    model: SPPPointerModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    grad_clip: float = 1.0,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            rect_feats = batch["rect_feats"].to(device)  # (B,N,F)
            rect_mask = batch["rect_mask"].to(device)    # (B,N)
            space_feat = batch["space_feat"].to(device)  # (B,F_space)
            step_idx = batch["step_idx"].to(device)      # (B,)
            targets = batch["targets"].to(device)        # (B,)

            opt.zero_grad(set_to_none=True)
            # Encode
            rect_enc, global_ctx = model.encode_rects(rect_feats, rect_mask)
            # Query
            space_emb = model.space_encoder(space_feat)
            step_emb = model.step_emb(step_idx)
            q = model.query_builder(space_emb, global_ctx, step_emb)  # (B,d)
            # Pointer scores
            q_exp = q.unsqueeze(1)  # (B,1,d)
            scores = torch.matmul(q_exp, rect_enc.transpose(1, 2)).squeeze(1) / math.sqrt(q.shape[-1])  # (B,N)
            scores = scores.masked_fill(~rect_mask, -1e9)

            loss = F.cross_entropy(scores, targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                preds = torch.argmax(scores, dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                total_loss += loss.item() * targets.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            val_acc = evaluate_pointer_model(model, val_loader, device)
            history["val_acc"].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            val_acc = float('nan')

        print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    return history


@torch.no_grad()
def evaluate_pointer_model(model: SPPPointerModel, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        rect_feats = batch["rect_feats"].to(device)
        rect_mask = batch["rect_mask"].to(device)
        space_feat = batch["space_feat"].to(device)
        step_idx = batch["step_idx"].to(device)
        targets = batch["targets"].to(device)

        rect_enc, global_ctx = model.encode_rects(rect_feats, rect_mask)
        space_emb = model.space_encoder(space_feat)
        step_emb = model.step_emb(step_idx)
        q = model.query_builder(space_emb, global_ctx, step_emb)
        q_exp = q.unsqueeze(1)
        scores = torch.matmul(q_exp, rect_enc.transpose(1, 2)).squeeze(1) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~rect_mask, -1e9)
        preds = torch.argmax(scores, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


# --------------------------------------------------
# Helper rápido para crear dataloaders
# --------------------------------------------------
def build_pointer_dataloaders(
    problems: List[List[Rect]],
    category: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
    teacher: Literal["fillratio", "hr", "hr_algo"] = "fillratio",
):
    steps = build_dataset_from_problems(problems, category, teacher=teacher)
    if not steps:
        raise ValueError("No se generaron pasos para el dataset pointer.")
    # Split
    indices = list(range(len(steps)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    val_count = int(len(indices) * val_split)
    val_idx = set(indices[:val_count])
    train_steps = [steps[i] for i in indices if i not in val_idx]
    val_steps = [steps[i] for i in indices if i in val_idx]

    train_ds = PointerStepsDataset(train_steps)
    val_ds = PointerStepsDataset(val_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pointer_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pointer_collate)
    return train_loader, val_loader, len(train_steps), len(val_steps)


if __name__ == "__main__":
    # Pequeño smoke test (usa problemas sintéticos simples)
    synthetic = [
        [(3,5),(4,4),(2,7),(5,2)],
        [(2,2),(3,3),(4,1),(1,4)],
        [(6,2),(2,6),(3,3),(1,5)],
    ]
    train_loader, val_loader, ntr, nv = build_pointer_dataloaders(synthetic, "C1", batch_size=4)
    print(f"Train steps: {ntr}  Val steps: {nv}")
    model = SPPPointerModel()
    history = train_pointer_model(model, train_loader, val_loader, epochs=2)
    print("History keys:", history.keys())
