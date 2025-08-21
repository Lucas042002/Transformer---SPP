from typing import List, Tuple, Dict, Optional

Space = Tuple[int, int, int, int]  # (x, y, w, h)
Rect  = Tuple[int, int]           # (w, h)

def _norm(v, den):
    return float(v) / float(den) if den else 0.0

def _rect_features_sin_seqid(rect: Rect, S: Space, W: int, Href: int) -> List[float]:
    """Features de rectángulo respecto del SUBESPACIO ACTIVO S=(x,y,w,h), sin seq_id."""
    w, h = rect
    _, _, Sw, Sh = S
    area   = w * h
    h_n    = _norm(h, Href)
    w_n    = _norm(w, W)
    area_n = _norm(area, W * Href)

    ratio     = (h / w) if w > 0 else 0.0
    max_den   = float(max(W, Href))
    max_side  = max(h, w) / max_den
    min_side  = min(h, w) / max_den

    fits     = (w <= Sw and h <= Sh)
    fits_rot = (h <= Sw and w <= Sh)

    if fits:
        slack_w = _norm(Sw - w, W)
        slack_h = _norm(Sh - h, Href)
        waste   = _norm((Sw * Sh) - area, W * Href)
    elif fits_rot:
        slack_w = _norm(Sw - h, W)
        slack_h = _norm(Sh - w, Href)
        waste   = _norm((Sw * Sh) - area, W * Href)
    else:
        slack_w, slack_h, waste = 0.0, 0.0, 1.0

    # <- sin seq_id
    return [
        h_n, 
        w_n, 
        area_n,
        0, #x_n
        0, #y_n
        # ratio, 
        # max_side, 
        # min_side,
        float(fits), 
        float(fits_rot),
        slack_w, 
        slack_h, 
        waste,
        0, # a_utilizar
        1  # type_id
    ]

def _space_features_sin_seqid(s: Space, W: int, Href: int, include_xy: bool = True, S_active: Space = None) -> List[float]:
    """Features de subespacio, sin seq_id."""
    x, y, w, h = s
    area   = w * h
    h_n    = _norm(h, Href)
    w_n    = _norm(w, W)
    area_n = _norm(area, W * Href)
    if include_xy:
        x_n = _norm(x, W)
        y_n = _norm(y, Href)
        return [
            h_n, 
            w_n, 
            area_n, 
            x_n, 
            y_n, 
            0, # fits
            0, # fits_rot
            0, # slack_w
            0, # slack_h
            0, # waste
            1 if S_active is not None and (x == S_active[0] and y == S_active[1] and w == S_active[2] and h == S_active[3]) else 0, # a_utilizar
            0, # type_id
        ]
    else:
        return [h_n, w_n, area_n]

def codificar_estado(
    spaces: List[Space],
    rects: List[Rect],
    espacio_seleccionado: Space,  # S ACTIVO de este paso
    W: int, 
    Href: int,
    include_xy: bool = True,
) -> Dict[str, object]:
    """
    Devuelve SOLO tensores de entrada y máscaras (sin y_rect, sin seq_id):
      - S_in:           (1, K, Fs)
      - S_mask:         (1, K)
      - R_in:           (1, N, Fr)
      - R_mask:         (1, N)
      - R_select_mask:  (1, N)
    """
    # Subespacios
    S_in, S_mask = [], []
    S_active = espacio_seleccionado

    for s in spaces:
        S_in.append(_space_features_sin_seqid(s, W, Href, include_xy=include_xy, S_active=S_active))
        S_mask.append(1)

    # Rectángulos
    R_in, R_mask, R_select_mask = [], [], []
    for r in rects:
        feats = _rect_features_sin_seqid(r, S_active, W, Href)
        R_in.append(feats)
        R_mask.append(1)
        fits_or = (feats[6] > 0.5) or (feats[7] > 0.5)  # indices 6,7 = fits, fits_rot
        R_select_mask.append(1 if fits_or else 0)

    # return {
    #     "S_in":           [S_in],          # (1, K, Fs)
    #     "S_mask":         [S_mask],        # (1, K)
    #     "R_in":           [R_in],          # (1, N, Fr)
    #     "R_mask":         [R_mask],        # (1, N)
    #     "R_select_mask":  [R_select_mask], # (1, N)
    # }
    return [
        S_in,        # (1, K, 12)
        R_in         # (1, N, 12)
    ]


def agregar_seq_id(samples):
    """
    Agrega seq_id al FINAL de cada vector en S_in y R_in, para todos los samples.
    seq_id = t / (T-1), donde t es el índice del paso, T = len(samples).
    Si T==1, seq_id=0.0.
    """
    T = len(samples)
    if T <= 1:
        seq_ids = [0.0] * T
    else:
        seq_ids = [t / (T - 1) for t in range(T)]

    for t, sample in enumerate(samples):
        sid = seq_ids[t]

        # Agregar seq_id al final de cada subespacio
        # sample["S_in"] tiene forma [S_in] -> (1, K, Fs); por eso iteramos sobre [0]
        for s in sample["S_in"][0]:
            s.append(sid)

        # Agregar seq_id al final de cada rectángulo
        for r in sample["R_in"][0]:
            r.append(sid)

    return samples


def agregar_seq_id_estados(all_states):
    """
    all_states: lista de pasos; cada paso es [S_in, R_in]
      - S_in: lista de vectores (subespacios)
      - R_in: lista de vectores (rectángulos), puede ser []
    Agrega/actualiza seq_id al FINAL de cada vector como t/(T-1).
    Devuelve una nueva lista (no muta la original).
    """
    import copy

    T = len(all_states)
    if T <= 1:
        seq_ids = [0.0] * T
    else:
        seq_ids = [t / (T - 1) for t in range(T)]

    out = copy.deepcopy(all_states)

    # Detectar longitud base de S y R (sin seq_id)
    base_len_S, base_len_R = None, None
    for step in out:
        # S_in
        if base_len_S is None and step and step[0]:
            for vec in step[0]:
                if isinstance(vec, (list, tuple)) and len(vec) > 0:
                    base_len_S = len(vec)
                    break
        # R_in
        if base_len_R is None and step and len(step) > 1 and step[1]:
            for vec in step[1]:
                if isinstance(vec, (list, tuple)) and len(vec) > 0:
                    base_len_R = len(vec)
                    break
        if base_len_S is not None and base_len_R is not None:
            break

    for t, step in enumerate(out):
        sid = seq_ids[t]

        # --- S_in ---
        if step and step[0]:
            for i, vec in enumerate(step[0]):
                if base_len_S is None:
                    # Si no se pudo inferir, intenta append directo
                    step[0][i] = list(vec) + [sid]
                else:
                    L = len(vec)
                    if L == base_len_S:
                        step[0][i] = list(vec) + [sid]
                    elif L == base_len_S + 1:
                        step[0][i][-1] = sid   # sobrescribe el seq_id existente
                    else:
                        # Longitud inesperada: aún así, agrega sid para no fallar
                        step[0][i] = list(vec) + [sid]

        # --- R_in ---
        if step and len(step) > 1 and step[1]:
            for i, vec in enumerate(step[1]):
                # Si nunca vimos R antes (todos vacíos hasta ahora), definimos base aquí
                if base_len_R is None:
                    base_len_R = len(vec)

                L = len(vec)
                if L == base_len_R:
                    step[1][i] = list(vec) + [sid]
                elif L == base_len_R + 1:
                    step[1][i][-1] = sid
                else:
                    step[1][i] = list(vec) + [sid]

    return out
