import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

class SPP2DTransformer(nn.Module):
    def __init__(self, state_dim, emb_dim, nhead, num_layers, dim_feedforward, num_actions, max_seq_len):
        super(SPP2DTransformer, self).__init__()
        self.state_embedding = nn.Linear(state_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(emb_dim, num_actions)
        self.value_head = nn.Linear(emb_dim, 1)

    def forward(self, state_seq):
        """
        state_seq: Tensor of shape (batch_size, seq_len, state_dim)
        """
        batch_size, seq_len, _ = state_seq.size()
        state_emb = self.state_embedding(state_seq)  # (batch_size, seq_len, emb_dim)
        pos_ids = torch.arange(seq_len, device=state_seq.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(pos_ids)
        x = state_emb + pos_emb
        x = self.transformer_encoder(x)
        # Usamos el embedding del último estado para las cabezas
        last_hidden = x[:, -1, :]
        policy_logits = self.policy_head(last_hidden)
        state_value = self.value_head(last_hidden)
        return policy_logits, state_value

# Ejemplo de uso:
if __name__ == "__main__":
    state_dim = 10      # Depende de cómo definiste el estado
    emb_dim = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    num_actions = 5     # Número de acciones posibles en SPP-2D
    max_seq_len = 20

    model = SPP2DTransformer(state_dim, emb_dim, nhead, num_layers, dim_feedforward, num_actions, max_seq_len)
    dummy_states = torch.randn(8, max_seq_len, state_dim)  # batch de 8 secuencias
    policy_logits, state_value = model(dummy_states)
    print(policy_logits.shape, state_value.shape)  # (8, num_actions), (8, 1)



def  pad_and_prepare_data(all_states, all_Y_rect):
    """
    Prepara los datos para el entrenamiento del modelo SPP-2D Transformer.
    
    Args:
        all_states: Lista de secuencias de estados.
        all_Y_rect: Lista de etiquetas correspondientes a las acciones.
    
    Returns:
        X: Tensor de estados preparados.
        Y: Tensor de etiquetas preparadas.
    """
    if all_states is None or all_Y_rect is None:
        raise ValueError("all_states and all_Y_rect must not be None")
    
    # Convertir a tensores y recortar/padear a una longitud máxima si es necesario
    max_seq_len = 20  # O el máximo que tengas en tus datos
    state_dim = len(all_states[0][0][0])  # Dimensión del vector de estado

    def pad_sequence(seq, max_len, pad_value=0):
        seq = seq[:max_len]
        while len(seq) < max_len:
            seq.append([pad_value]*state_dim)
        return seq

    def pad_label(seq, max_len, pad_value=0):
        seq = seq[:max_len]
        while len(seq) < max_len:
            seq.append([pad_value]*len(seq[0]))
        return seq

    X = [pad_sequence(s, max_seq_len) for s in all_states]
    Y = [pad_label(y, max_seq_len) for y in all_Y_rect]

    X = torch.tensor(X, dtype=torch.float32)  # (num_samples, max_seq_len, state_dim)
    Y = torch.tensor(Y, dtype=torch.float32)  # (num_samples, max_seq_len, num_actions)

    return X, Y
def create_model(state_dim, num_actions, max_seq_len):
    """ Crea una instancia del modelo SPP-2D Transformer.
    Args:
        state_dim: Dimensión del vector de estado.
        num_actions: Número de acciones posibles.
        max_seq_len: Longitud máxima de las secuencias de entrada.
    Returns:
        model: Instancia del modelo SPP-2D Transformer.
    """
    emb_dim = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    model = SPP2DTransformer(state_dim, emb_dim, nhead, num_layers, dim_feedforward, num_actions, max_seq_len)
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, max_seq_len):
    """ Entrena el modelo SPP-2D Transformer con los datos proporcionados.
    Args:
        model: Instancia del modelo SPP-2D Transformer.
        X_train: Tensor de estados de entrenamiento.
        Y_train: Tensor de etiquetas de entrenamiento.
        X_val: Tensor de estados de validación.
        Y_val: Tensor de etiquetas de validación.
        max_seq_len: Longitud máxima de las secuencias de entrada.
    """
    num_actions = Y_train.shape[-1]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()  # Para one-hot, o CrossEntropyLoss si usas índices

    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        policy_logits, _ = model(X_train)
        # Solo usamos la última acción de la secuencia para entrenamiento supervisado
        loss = criterion(policy_logits, Y_train[:, -1, :])  # O ajusta según tu formato de Y
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


def validate_model(model, X_val, Y_val):
    """ Valida el modelo SPP-2D Transformer con los datos de validación.
    Args:
        model: Instancia del modelo SPP-2D Transformer.
        X_val: Tensor de estados de validación.
        Y_val: Tensor de etiquetas de validación.
    Returns:
        accuracy: Precisión del modelo en el conjunto de validación.
    """
    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(X_val)
        preds = torch.sigmoid(policy_logits)
        correct = (preds.argmax(dim=1) == Y_val[:, -1, :].argmax(dim=1)).float().mean()
        return correct.item()
    