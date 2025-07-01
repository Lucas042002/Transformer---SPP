import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



# Clase para la capa de salida final
class FinalOutputLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FinalOutputLayer, self).__init__()
        self.output_projection = nn.Linear(input_dim, num_classes)

    def forward(self, x, return_probabilities=False):
        # Proyección de salida
        output = self.output_projection(x)  # [batch_size, seq_length, num_classes]
        if return_probabilities:
            return F.softmax(output, dim=-1)
        else:
            return output

class CustomModel(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, num_layers=2, dropout_rate=0.2, num_classes=10):
        super(CustomModel, self).__init__()
        #self.seq_length = seq_length  # Asumiendo una longitud fija de secuencia para simplificar
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Proyección de entrada
        self.input_projection = nn.Linear(input_dim, num_heads * head_dim)

        # Crear múltiples capas de atención y densa
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'multihead_attention': nn.MultiheadAttention(embed_dim=num_heads * head_dim, num_heads=num_heads, dropout=dropout_rate),
                'dense_layer': nn.Sequential(
                    nn.Linear(num_heads * head_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_heads * head_dim)
                ),
                'norm1': nn.LayerNorm(num_heads * head_dim),
                'norm2': nn.LayerNorm(num_heads * head_dim)
            }) for _ in range(num_layers)
        ])

        # Capa de salida final
        self.final_output_layer = FinalOutputLayer(num_heads * head_dim, num_classes)
    
    def generate_attention_mask(self, x, num_heads, padding_value=0):
        # Identificar posiciones de padding en x
        mask = (x.sum(dim=-1) == padding_value)  # [batch_size, seq_length]
        mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)  # Expandir a [batch_size, seq_length, seq_length]
        mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)  # Expandir para incluir num_heads: [batch_size, num_heads, seq_length, seq_length]
        mask = mask.reshape(-1, x.size(1), x.size(1))  # Ajustar a [batch_size * num_heads, seq_length, seq_length]
        mask = mask.to(dtype=torch.bool)  # Convertir a bool para usar como máscara
        return mask


    def forward(self, x, seq_lengths=10, return_probabilities=False):
        # x: [batch_size, seq_length, input_dim]
        x = x.float()

        # Proyección de entrada
        x_proj = self.input_projection(x)

        # Generar máscara de atención
        attn_mask = self.generate_attention_mask(x, self.num_heads)

        # Aplicar cada capa de atención y densa
        for layer in self.layers:
            x_proj = x_proj.permute(1, 0, 2)
            attn_output, _ = layer['multihead_attention'](x_proj, x_proj, x_proj)
            attn_output = attn_output.permute(1, 0, 2)
            x_proj = x_proj.permute(1, 0, 2)
            attn_output = layer['norm1'](attn_output + x_proj)
            dense_output = layer['dense_layer'](attn_output)
            x_proj = layer['norm2'](dense_output + attn_output)

        # Pooling: usa el primer paso (puedes cambiar a mean o last)
        pooled = x_proj[:, 0, :]  # [batch_size, hidden_dim]
        output = self.final_output_layer.output_projection(pooled)  # [batch_size, num_classes]
        if return_probabilities:
            return F.softmax(output, dim=-1)
        else:
            return output
    

def procesar_datos_entrada(largo_max, all_states, all_Y_rect, verbose=False):
    
    for i, all_state in enumerate(all_states):
        for j, state in enumerate(all_state):
            if len(state) < largo_max:
                # Rellenar con [0,0,0,0,0] hasta alcanzar largo_max
                state += [[0, 0, 0, 0, 0]] * (largo_max - len(state))

    # Asegurarse de que todas las acciones tengan el mismo largo
    for i, state in enumerate(all_Y_rect):
        for j, action in enumerate(state):
            if len(action) < largo_max:
                # Rellenar con 0 hasta alcanzar largo_max
                action += [0] * (largo_max - len(action))

    # Filtrar estados y acciones donde all_Y_rect es solo ceros
    all_states_filtrado = []
    all_Y_rect_filtrado = []
    for estados, acciones in zip(all_states, all_Y_rect):
        nuevos_estados = []
        nuevas_acciones = []
        for estado, accion in zip(estados, acciones):
            if not (isinstance(accion, list) and all(a == 0 for a in accion)):
                nuevos_estados.append(estado)
                nuevas_acciones.append(accion)
        if nuevos_estados:  # Solo agregar si hay al menos un estado válido
            all_states_filtrado.append(nuevos_estados)
            all_Y_rect_filtrado.append(nuevas_acciones)

    # Usa los filtrados para el resto del procesamiento
    all_states = all_states_filtrado
    all_Y_rect = all_Y_rect_filtrado

    X = []
    Y = []
    for estados, acciones in zip(all_states, all_Y_rect):
        for state, action in zip(estados, acciones):
            X.append(state)
            # Convierte one-hot a índice (o 0 si todo es cero)
            if isinstance(action, list) or isinstance(action, np.ndarray):
                if np.sum(action) == 0:
                    Y.append(0)
                else:
                    Y.append(int(np.argmax(action)))
            else:
                Y.append(int(action))

    X = np.array(X)  # shape: (num_samples, largo_max, 5)

    X = X.astype(np.float32)
    
    for i in range(X.shape[-1]):
        max_val = np.abs(X[..., i]).max()
        if max_val > 0:
            X[..., i] /= max_val

    Y = np.array(Y)  # shape: (num_samples,)

    if verbose:
        print(np.bincount(Y))  # Y es tu vector de etiquetas

    if verbose:
        print(f"Total de secuencias X: {len(X)}")
        print(f"Total de etiquetas Y: {len(Y)}")
        # print("Primeros 5 X:", X[:5])
        # print("Primeros 5 Y:", Y[:5])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    # División entrenamiento/validación
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

    batch_size = 32
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, X_tensor.shape[1], 1  # output_seq_length=1

def entrenamiento(model, train_loader, val_loader, optimizer, criterion, epochs=50, categoria=None):
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        test = 'debug3'  # Cambiar a 'debug' para imprimir información de depuración
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)  # [batch, num_classes]
            loss = criterion(logits, yb)  # yb: [batch]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Validación
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = logits.argmax(dim=-1)  # [batch]
                mask = (yb != 0)
                aciertos = (preds == yb) & mask
                correct += aciertos.sum().item()
                total += mask.sum().item()
                if test == 'debug':
                    print(f"  Validación - Epoch {epoch+1}/{epochs}")
                    print(f"  xb {xb.shape} ")
                    print(f"  yb {yb.shape} {yb}")
                    print(f"  preds {preds.shape} {preds}")
                    print(f"  logits {logits.shape} {logits}")
                    test = 'debug2'

        acc = correct / total if total > 0 else 0
        val_accuracies.append(acc)
        print(f"  Validación accuracy (solo relevantes): {acc:.4f}")

    torch.save(model.state_dict(), f'SPP_transformer_{categoria}.pth')

    # Visualización de la curva de pérdida y accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, marker='o')
    plt.title('Curva de pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')

    plt.subplot(1,2,2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Accuracy de validación')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(f'training_curves_{categoria}.png')
    plt.close()



