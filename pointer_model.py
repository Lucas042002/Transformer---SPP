import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RectEncoder(nn.Module):
    """Encoder que procesa todos los rectángulos una sola vez (cacheable).
    rect_feats: (B, N, F)
    rect_mask: (B, N) boolean True = disponible / válido
    Devuelve: (B, N, d_model)
    """
    def __init__(self, 
                 d_model: int, 
                 input_dim: int, 
                 num_layers: int = 2, 
                 num_heads: int = 4, 
                 d_ff: int = 256, 
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, rect_feats: torch.Tensor, rect_mask: torch.Tensor | None = None):
        x = self.proj(rect_feats)
        # En PyTorch, src_key_padding_mask = True indica PAD -> ignorar. Queremos lo contrario de rect_mask (True = válido)
        pad_mask = None
        if rect_mask is not None:
            pad_mask = ~rect_mask  # (B,N)
        return self.encoder(x, src_key_padding_mask=pad_mask)  # (B,N,d)


class SpaceEncoder(nn.Module):
    """Proyección simple de features del espacio activo."""
    def __init__(self, d_model: int, space_dim: int):
        super().__init__()
        self.proj = nn.Linear(space_dim, d_model)

    def forward(self, space_feat: torch.Tensor):  # (B, space_dim)
        return self.proj(space_feat)  # (B,d)


class StepEmbedding(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_steps, d_model)

    def forward(self, step_idx: torch.Tensor):  # (B,)
        return self.emb(step_idx)  # (B,d)


class QueryBuilder(nn.Module):
    """Construye el query a partir de (space_emb, global_ctx, step_emb)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, space_emb: torch.Tensor, global_ctx: torch.Tensor, step_emb: torch.Tensor):
        # Todas (B,d)
        q = torch.cat([space_emb, global_ctx, step_emb], dim=-1)
        return self.ff(q)  # (B,d)


class PointerDecoderStep(nn.Module):
    """Un solo paso: atención tipo pointer sobre embeddings de rectángulos.
    rect_enc: (B,N,d)  rect_mask: (B,N) True = disponible
    q: (B,d)
    Devuelve probs (B,N) y raw scores (B,N)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = math.sqrt(d_model)

    def forward(self, q: torch.Tensor, rect_enc: torch.Tensor, rect_mask: torch.Tensor):
        q = q.unsqueeze(1)  # (B,1,d)
        scores = torch.matmul(q, rect_enc.transpose(1, 2)).squeeze(1) / self.scale  # (B,N)
        scores = scores.masked_fill(~rect_mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        return probs, scores


class SPPPointerModel(nn.Module):
    """Modelo pointer minimal para experimentación.
    Uso:
      enc_rects, global_ctx = model.encode_rects(rect_feats, rect_mask)
      probs, scores = model.decode_step(enc_rects, rect_mask, space_feat, step_idx)
    """
    def __init__(self, d_model: int = 256, rect_feat_dim: int = 12, space_feat_dim: int = 12, num_enc_layers: int = 2,
                 num_heads: int = 4, d_ff: int = 512, dropout: float = 0.1, max_steps: int = 256):
        super().__init__()
        self.rect_encoder = RectEncoder(d_model, rect_feat_dim, num_enc_layers, num_heads, d_ff, dropout)
        self.space_encoder = SpaceEncoder(d_model, space_feat_dim)
        self.step_emb = StepEmbedding(d_model, max_steps)
        self.query_builder = QueryBuilder(d_model)
        self.pointer = PointerDecoderStep(d_model)
        self.global_linear = nn.Linear(d_model, d_model)

    @torch.no_grad()
    def encode_rects(self, rect_feats: torch.Tensor, rect_mask: torch.Tensor):
        enc = self.rect_encoder(rect_feats, rect_mask)  # (B,N,d)
        mask_f = rect_mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_ctx = (enc * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,d)
        global_ctx = self.global_linear(global_ctx)
        return enc, global_ctx

    @torch.no_grad()
    def decode_step(self, rect_enc: torch.Tensor, rect_mask: torch.Tensor, space_feat: torch.Tensor, step_idx: torch.Tensor,
                    global_ctx: torch.Tensor):
        space_emb = self.space_encoder(space_feat)  # (B,d)
        step_embedding = self.step_emb(step_idx)    # (B,d)
        q = self.query_builder(space_emb, global_ctx, step_embedding)  # (B,d)
        probs, scores = self.pointer(q, rect_enc, rect_mask)
        return probs, scores


def ejemplo_inferencia():
    """Ejemplo rápido de uso aislado para validar estructura."""
    B, N, F_rect, F_space = 1, 5, 12, 12
    rect_feats = torch.rand(B, N, F_rect)
    rect_mask = torch.ones(B, N, dtype=torch.bool)
    space_feat = torch.rand(B, F_space)
    model = SPPPointerModel()
    rect_enc, g = model.encode_rects(rect_feats, rect_mask)
    probs, scores = model.decode_step(rect_enc, rect_mask, space_feat, torch.tensor([0]), g)
    print("Probs:", probs)
    return probs

if __name__ == "__main__":  # Permite ejecutar standalone
    ejemplo_inferencia()
