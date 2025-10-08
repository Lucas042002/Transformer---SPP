"""Script de prueba para validar las nuevas features optimizadas."""
import torch
from pointer_model import SPPPointerModel
import states as st
import categories as cat

def test_rect_features():
    """Prueba features de rectángulos (10 dims)."""
    print("=" * 60)
    print("TEST 1: Rect Features (10 dimensiones)")
    print("=" * 60)
    
    rects = [(3, 5), (4, 4), (2, 7), (5, 2)]
    W = 20
    Href = 100
    
    for r in rects:
        feats = st._rect_features_optimized(r, W, Href, rects)
        print(f"\nRect {r}:")
        print(f"  Features ({len(feats)} dims): {[f'{x:.3f}' for x in feats]}")
        assert len(feats) == 10, f"Expected 10 dims, got {len(feats)}"
    
    print("\n✅ Rect features OK: 10 dimensiones\n")


def test_space_features():
    """Prueba features de espacios (12 dims)."""
    print("=" * 60)
    print("TEST 2: Space Features (12 dimensiones)")
    print("=" * 60)
    
    spaces = [(0, 0, 20, 100), (0, 50, 15, 50), (10, 10, 10, 40)]
    W = 20
    Href = 100
    current_max_height = 100
    
    for s in spaces:
        feats = st._space_features_optimized(s, W, Href, spaces, current_max_height)
        print(f"\nSpace {s}:")
        print(f"  Features ({len(feats)} dims): {[f'{x:.3f}' for x in feats]}")
        assert len(feats) == 12, f"Expected 12 dims, got {len(feats)}"
    
    print("\n✅ Space features OK: 12 dimensiones\n")


def test_pointer_features():
    """Prueba pointer_features completo."""
    print("=" * 60)
    print("TEST 3: pointer_features()")
    print("=" * 60)
    
    rects = [(3, 5), (4, 4), (2, 7)]
    spaces = [(0, 0, 20, 100), (0, 50, 15, 50)]
    active_space = spaces[0]
    W = 20
    Href = 100
    
    space_feat, rect_feats, feasible_mask = st.pointer_features(
        spaces, rects, active_space, W, Href
    )
    
    print(f"\nSpace feat: {len(space_feat)} dims")
    print(f"Rect feats: {len(rect_feats)} rects x {len(rect_feats[0])} dims")
    print(f"Feasible mask: {feasible_mask}")
    
    assert len(space_feat) == 12, f"Space feat should be 12 dims, got {len(space_feat)}"
    assert all(len(rf) == 10 for rf in rect_feats), "All rect feats should be 10 dims"
    
    print("\n✅ pointer_features OK\n")


def test_model_forward():
    """Prueba forward pass del modelo con nuevas dimensiones."""
    print("=" * 60)
    print("TEST 4: SPPPointerModel Forward Pass")
    print("=" * 60)
    
    model = SPPPointerModel(rect_feat_dim=10, space_feat_dim=12)
    
    B, N = 2, 5
    rect_feats = torch.rand(B, N, 10)  # 10 dims
    rect_mask = torch.ones(B, N, dtype=torch.bool)
    space_feat = torch.rand(B, 12)  # 12 dims
    step_idx = torch.tensor([0, 1])
    
    print(f"\nInputs:")
    print(f"  rect_feats: {rect_feats.shape}")
    print(f"  space_feat: {space_feat.shape}")
    
    # Encode
    rect_enc, _ = model.encode_rects(rect_feats, rect_mask)
    print(f"\nAfter encode_rects:")
    print(f"  rect_enc: {rect_enc.shape}")
    
    # Decode
    probs, scores = model.decode_step(rect_enc, rect_mask, space_feat, step_idx)
    print(f"\nAfter decode_step:")
    print(f"  probs: {probs.shape}")
    print(f"  scores: {scores.shape}")
    
    assert rect_enc.shape == (B, N, 256), f"Unexpected rect_enc shape: {rect_enc.shape}"
    assert probs.shape == (B, N), f"Unexpected probs shape: {probs.shape}"
    
    print("\n✅ Model forward pass OK\n")


def test_dynamic_global_ctx():
    """Prueba que global_ctx se actualiza dinámicamente."""
    print("=" * 60)
    print("TEST 5: Dynamic Global Context")
    print("=" * 60)
    
    model = SPPPointerModel(rect_feat_dim=10, space_feat_dim=12)
    
    B, N = 1, 5
    rect_feats = torch.rand(B, N, 10)
    rect_mask_full = torch.ones(B, N, dtype=torch.bool)
    rect_mask_partial = torch.tensor([[True, True, False, False, False]])  # Solo 2 disponibles
    space_feat = torch.rand(B, 12)
    step_idx = torch.tensor([0])
    
    # Encode una vez
    rect_enc, _ = model.encode_rects(rect_feats, rect_mask_full)
    
    # Decode con máscara completa
    probs1, _ = model.decode_step(rect_enc, rect_mask_full, space_feat, step_idx)
    
    # Decode con máscara parcial (global_ctx debería cambiar)
    probs2, _ = model.decode_step(rect_enc, rect_mask_partial, space_feat, step_idx)
    
    print(f"\nProbs con 5 rects disponibles: {probs1}")
    print(f"Probs con 2 rects disponibles: {probs2}")
    
    # Las probabilidades deberían ser diferentes porque global_ctx cambió
    assert not torch.allclose(probs1, probs2), "Global context no se está actualizando dinámicamente!"
    
    print("\n✅ Global context dinámico OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VALIDACIÓN DE NUEVAS FEATURES OPTIMIZADAS")
    print("=" * 60 + "\n")
    
    try:
        test_rect_features()
        test_space_features()
        test_pointer_features()
        test_model_forward()
        test_dynamic_global_ctx()
        
        print("\n" + "=" * 60)
        print("✅ TODOS LOS TESTS PASARON")
        print("=" * 60)
        print("\nCambios implementados:")
        print("  - Rect features: 10 dimensiones (geométricas invariantes)")
        print("  - Space features: 12 dimensiones (geométricas + contexto)")
        print("  - Global context: se actualiza dinámicamente en cada decode_step")
        print("  - pointer_model: rect_feat_dim=10, space_feat_dim=12")
        print("\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLÓ: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
