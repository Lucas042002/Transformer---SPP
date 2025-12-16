"""Entrenamiento del modelo pointer mediante Imitation Learning.

El modelo aprende a imitar las decisiones del algoritmo HR que
actúa como "maestro experto". En cada paso de empaquetamiento, el modelo observa:
  - Features de todos los rectángulos disponibles (10 dims cada uno)
  - Features del espacio activo actual (19 dims)
  - Contexto global dinámico (qué rectángulos quedan)

Y aprende a predecir qué rectángulo eligió el HR en esa situación.

Arquitectura de aprendizaje:
  1. Generar trayectorias usando hr_algorithm.hr_packing_pointer (maestro experto)
     → Devuelve directamente PointerSteps con features optimizados
  2. Cada PointerStep contiene: (rect_feats, rect_mask, space_feat, target, step_idx)
  3. Entrenar modelo pointer con cross-entropy sobre las acciones del maestro

Nota: hr_packing_pointer() genera los PointerSteps directamente durante la ejecución
      del algoritmo HR, sin necesidad de re-simular el proceso.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal
import math
import random
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import categories as cat
import states as st
import hr_algorithm as hr
from hr_algorithm import PointerStep 
from pointer_model import SPPPointerModel

Rect = Tuple[int, int]
Space = Tuple[int, int, int, int]


# --------------------------------------------------
# Generación de trayectorias desde HR Algorithm
# --------------------------------------------------
def build_dataset_from_problems(
    problems: List[List[Rect]],
    category: str,
    augment_permutations: int = 0
) -> List[PointerStep]:
    """Genera dataset de entrenamiento con data augmentation opcional.
    
    Usa hr_algorithm.hr_packing_pointer() que ejecuta el HR y devuelve directamente
    los PointerSteps con features optimizados (10 dims rect, 19 dims space).
    
    Args:
        problems: Lista de problemas, donde cada problema es una lista de rectángulos
        category: Categoría del problema (determina W y Href)
        augment_permutations: Número de permutaciones aleatorias adicionales por problema (default: 0)
                             Si es 0, solo usa el orden óptimo del HR
                             Si es >0, genera N versiones con orden aleatorio adicional
        
    Returns:
        Lista plana de PointerStep (todos los pasos de todos los problemas concatenados)
        Con augmentation: total_steps = len(problems) * (1 + augment_permutations) * steps_per_problem
    """
    out: List[PointerStep] = []
    W = cat.CATEGORIES[category]["width"]
    
    for idx, rects in enumerate(problems):
        # VERSIÓN ORIGINAL: Orden óptimo del HR (con permutaciones internas)
        pointer_steps, placements, altura, rect_sequence, Y_rect = hr.heuristic_recursion_pointer(
            rects=rects,
            container_width=W,
            category=category
        )
        
        out.extend(pointer_steps)
        
        if len(pointer_steps) != cat.CATEGORIES[category]["num_items"]:
            print(f"  Problema {idx+1}: Se esperaban {cat.CATEGORIES[category]['num_items']} pasos, pero se generaron {len(pointer_steps)}")
        
        # DATA AUGMENTATION: Versiones con órdenes aleatorios
        if augment_permutations > 0:
            for aug_idx in range(augment_permutations):
                # Permutar aleatoriamente los rectángulos
                rects_shuffled = rects.copy()
                random.shuffle(rects_shuffled)
                
                # Generar trayectoria con este orden aleatorio
                pointer_steps_aug, _, _, _, _ = hr.heuristic_recursion_pointer(
                    rects=rects_shuffled,
                    container_width=W,
                    category=category
                )
                
                out.extend(pointer_steps_aug)
    
    total_problemas = len(problems)
    problemas_originales = total_problemas
    problemas_augmentados = total_problemas * augment_permutations
    
    if augment_permutations > 0:
        print(f"\nDataset con Data Augmentation:")
        print(f"  Problemas originales: {problemas_originales}")
        print(f"  Problemas augmentados: {problemas_augmentados} ({augment_permutations} permutaciones x {total_problemas})")
        print(f"  Total problemas: {problemas_originales + problemas_augmentados}")
        print(f"  Total pasos: {len(out)}")
    else:
        print(f"\nDataset completo: {len(out)} pasos de {len(problems)} problemas")
    
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
    categoria: str = "C1",
    guardar_modelo: bool = True,
    num_enc_layers: int = None,
    num_heads: int = None,
    early_stopping_patience: int = 10,
    use_weighted_loss: bool = True, 
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    
    # Crear pesos para pasos (pasos finales pesan más)
    max_steps = cat.CATEGORIES[categoria]["num_items"]
    if use_weighted_loss:
        step_weights = torch.linspace(1.0, 2.0, steps=max_steps, device=device)
        print(f"Usando weighted loss: pasos iniciales (peso=1.0), pasos finales (peso=2.0)")
    else:
        step_weights = None

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
            
            # Encode rectángulos (una vez)
            rect_enc, _ = model.encode_rects(rect_feats, rect_mask)
            
            # Recalcular global_ctx DINÁMICAMENTE 
            mask_f = rect_mask.float()
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            current_global_ctx = (rect_enc * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)
            current_global_ctx = model.global_linear(current_global_ctx)
            
            # Query builder con global_ctx DINÁMICO
            space_emb = model.space_encoder(space_feat)
            step_emb = model.step_emb(step_idx)
            q = model.query_builder(space_emb, current_global_ctx, step_emb)  # (B,d)
            
            # Pointer scores
            q_exp = q.unsqueeze(1)  # (B,1,d)
            scores = torch.matmul(q_exp, rect_enc.transpose(1, 2)).squeeze(1) / math.sqrt(q.shape[-1])  # (B,N)
            scores = scores.masked_fill(~rect_mask, -1e9)

            # Loss con pesos opcionales
            if step_weights is not None:
                # Aplicar pesos por paso
                loss = F.cross_entropy(scores, targets, reduction='none')
                weights = step_weights[step_idx]  # (B,)
                loss = (loss * weights).mean()
            else:
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
                best_epoch = epoch
                epochs_without_improvement = 0
                print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} NEW BEST!")
            else:
                epochs_without_improvement += 1
                print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} (no improvement: {epochs_without_improvement}/{early_stopping_patience})")
                
                # Early stopping check
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEARLY STOPPING: No mejora en {early_stopping_patience} épocas consecutivas")
                    print(f"   Mejor accuracy: {best_val_acc:.4f} en época {best_epoch}")
                    break
        else:
            val_acc = float('nan')
            print(f"[Epoch {epoch:02d}] loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    # Guardar modelo y generar visualización
    if guardar_modelo:
        guardar_modelo_pointer(model, opt, history, epochs, categoria, num_enc_layers, num_heads)
        guardar_imagen_entrenamiento_pointer(
            history["train_loss"], 
            history["train_acc"], 
            history["val_acc"], 
            categoria,
            num_enc_layers,
            num_heads
        )

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

        # Encode rectángulos
        rect_enc, _ = model.encode_rects(rect_feats, rect_mask)
        
        # Recalcular global_ctx DINÁMICAMENTE basado en rect_mask actual
        mask_f = rect_mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        current_global_ctx = (rect_enc * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        current_global_ctx = model.global_linear(current_global_ctx)
        
        # Query builder con global_ctx DINÁMICO
        space_emb = model.space_encoder(space_feat)
        step_emb = model.step_emb(step_idx)
        q = model.query_builder(space_emb, current_global_ctx, step_emb)
        
        q_exp = q.unsqueeze(1)
        scores = torch.matmul(q_exp, rect_enc.transpose(1, 2)).squeeze(1) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~rect_mask, -1e9)
        preds = torch.argmax(scores, dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / max(total, 1)


# --------------------------------------------------
# Funciones de visualización y guardado
# --------------------------------------------------
def guardar_imagen_entrenamiento_pointer(train_losses, train_accs, val_accs, categoria, 
                                         num_enc_layers=None, num_heads=None):
    """
    Guarda una imagen con las curvas de entrenamiento del modelo pointer
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento por época
        train_accs: Lista de accuracies de entrenamiento por época
        val_accs: Lista de accuracies de validación por época
        categoria: Categoría del problema (e.g., "C1")
        num_enc_layers: Número de capas del encoder (opcional)
        num_heads: Número de attention heads (opcional)
    """
    # Crear carpeta img si no existe
    os.makedirs("img", exist_ok=True)
    
    # Calcular accuracy máxima
    max_val_acc = max(val_accs) if val_accs else 0.0
    max_train_acc = max(train_accs) if train_accs else 0.0
    
    # Crear la figura con 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Gráfico de pérdida
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title(f'Training Loss - Pointer {categoria}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico de train accuracy
    ax2.plot(train_accs, 'g-', label='Training Accuracy', linewidth=2)
    ax2.set_title(f'Training Accuracy - Pointer {categoria}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # Gráfico de validation accuracy
    ax3.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax3.set_title(f'Validation Accuracy - Pointer {categoria}', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    # Añadir información adicional
    final_loss = train_losses[-1] if train_losses else 0
    
    # Construir título con hiperparámetros si están disponibles
    hyperparams_str = ""
    if num_enc_layers is not None and num_heads is not None:
        hyperparams_str = f' | Layers: {num_enc_layers} | Heads: {num_heads}'
    
    fig.suptitle(f'Pointer Model Training Results - {categoria}{hyperparams_str}\n'
                f'Final Loss: {final_loss:.4f} | Train Acc: {max_train_acc:.4f} | Val Acc: {max_val_acc:.4f}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Nombre con accuracy y hiperparámetros
    accuracy_str = f"{max_val_acc:.4f}".replace("0.", "")  # 0.8745 -> 8745
    
    # Construir nombre con hiperparámetros
    hyperparams_suffix = ""
    if num_enc_layers is not None and num_heads is not None:
        hyperparams_suffix = f"_L{num_enc_layers}_H{num_heads}"
    
    image_filename = f"img/pointer_{categoria.lower()}{hyperparams_suffix}_acc{accuracy_str}.png"
    
    plt.savefig(image_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\nImagen de entrenamiento guardada en: {image_filename}")


def guardar_modelo_pointer(model, optimizer, history, epochs, categoria,
                           num_enc_layers=None, num_heads=None):
    """
    Guarda el modelo pointer entrenado con accuracy máxima e hiperparámetros en el nombre
    
    Args:
        model: Modelo SPPPointerModel entrenado
        optimizer: Optimizador usado
        history: Diccionario con métricas de entrenamiento
        epochs: Número de épocas entrenadas
        categoria: Categoría del problema (e.g., "C1")
        num_enc_layers: Número de capas del encoder 
        num_heads: Número de attention heads 
    """
    # Crear carpeta models si no existe
    os.makedirs("models", exist_ok=True)
    
    # Calcular accuracy máxima de validación
    max_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
    
    # Nombre del archivo con accuracy y hiperparámetros
    accuracy_str = f"{max_val_acc:.4f}".replace("0.", "")  # 0.8745 -> 8745
    
    # Construir nombre con hiperparámetros
    hyperparams_suffix = ""
    if num_enc_layers is not None and num_heads is not None:
        hyperparams_suffix = f"_L{num_enc_layers}_H{num_heads}"
    
    model_filename = f"models/pointer_{categoria.lower()}{hyperparams_suffix}_acc{accuracy_str}.pth"
    
    # Guardar el modelo completo con hiperparámetros
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'max_val_acc': max_val_acc,
        'categoria': categoria,
        'num_enc_layers': num_enc_layers,
        'num_heads': num_heads,
    }, model_filename)
    
    print(f"Modelo guardado en: {model_filename}")
    print(f"Accuracy máxima de validación: {max_val_acc:.4f}")
    if num_enc_layers is not None and num_heads is not None:
        print(f"Hiperparámetros: {num_enc_layers} layers, {num_heads} heads")


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
    augment_permutations: int = 0,
):
    """Construye dataloaders de entrenamiento y validación con data augmentation opcional.
    
    Args:
        problems: Lista de problemas (cada uno es una lista de rectángulos)
        category: Categoría del problema
        batch_size: Tamaño de batch
        val_split: Fracción de datos para validación
        shuffle: Si mezclar los datos antes de split
        seed: Semilla para reproducibilidad
        augment_permutations: Número de permutaciones aleatorias adicionales por problema (default: 0)
        
    Returns:
        train_loader, val_loader, num_train_steps, num_val_steps
    """
    print(f"\n{'='*80}")
    print(f"CONSTRUYENDO DATALOADERS")
    print(f"{'='*80}")
    print(f"Categoría: {category}")
    print(f"Problemas: {len(problems)}")
    print(f"Batch size: {batch_size}")
    print(f"Val split: {val_split}")
    if augment_permutations > 0:
        print(f"Data Augmentation: {augment_permutations} permutaciones adicionales por problema")
    
    # Generar todos los pasos de todos los problemas (con augmentation si está activo)
    steps = build_dataset_from_problems(problems, category, augment_permutations)
    if not steps:
        raise ValueError("No se generaron pasos para el dataset pointer.")
    
    print(f"\nESTADÍSTICAS DEL DATASET:")
    print(f"  Total de pasos: {len(steps)}")
    
    # IMPORTANTE: Con augmentation, cada problema genera (1 + augment_permutations) variantes
    # Por ejemplo: 500 problemas con augment_permutations=2 → 1500 variantes totales
    total_problem_variants = len(problems) * (1 + augment_permutations)
    steps_per_variant = cat.CATEGORIES[category]["num_items"]
    
    print(f"  Problemas originales: {len(problems)}")
    print(f"  Variantes totales: {total_problem_variants} (incluye augmentation)")
    print(f"  Pasos por variante: {steps_per_variant}")
    
    # Split train/val a nivel de PROBLEMAS ORIGINALES (no variantes)
    # Esto asegura que todas las variantes augmentadas de un problema estén juntas
    problem_indices = list(range(len(problems)))
    if shuffle:
        random.Random(seed).shuffle(problem_indices)
    
    val_problem_count = max(1, int(len(problems) * val_split))
    val_problem_idx = set(problem_indices[:val_problem_count])
    
    print(f"\nSPLIT TRAIN/VAL:")
    print(f"  Problemas de validación: {val_problem_count}/{len(problems)}")
    
    # Asignar pasos a train o val según su problema de origen
    train_steps = []
    val_steps = []
    
    step_idx = 0
    for prob_idx in range(len(problems)):
        # Cada problema genera steps_per_variant pasos × (1 + augment_permutations) variantes
        num_steps_for_problem = steps_per_variant * (1 + augment_permutations)
        problem_steps = steps[step_idx:step_idx + num_steps_for_problem]
        
        if prob_idx in val_problem_idx:
            val_steps.extend(problem_steps)
        else:
            train_steps.extend(problem_steps)
        
        step_idx += num_steps_for_problem
    
    print(f"  Pasos de entrenamiento: {len(train_steps)}")
    print(f"  Pasos de validación: {len(val_steps)}")

    train_ds = PointerStepsDataset(train_steps)
    val_ds = PointerStepsDataset(val_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pointer_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pointer_collate)
    
    print(f"\nDataloaders creados:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"{'='*80}\n")
    
    return train_loader, val_loader, len(train_steps), len(val_steps)


