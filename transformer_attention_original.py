import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
import math 
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.pos_embed_matrix = torch.zeros(max_seq_len, d_model, device=device)
        token_pos = torch.arange(0, max_seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() 
                             * (-math.log(10000.0)/d_model))
        self.pos_embed_matrix[:, 0::2] = torch.sin(token_pos * div_term)
        self.pos_embed_matrix[:, 1::2] = torch.cos(token_pos * div_term)
        self.pos_embed_matrix = self.pos_embed_matrix.unsqueeze(0).transpose(0,1)
        
    def forward(self, x):
#         print(self.pos_embed_matrix.shape)
#         print(x.shape)
        return x + self.pos_embed_matrix[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model = 512, num_heads = 8):
        super().__init__()
        assert d_model % num_heads == 0, 'Embedding size not compatible with num heads'
        
        self.d_v = d_model // num_heads
        self.d_k = self.d_v
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask = None):
        batch_size = Q.size(0)
        '''
        Q, K, V -> [batch_size, seq_len, num_heads*d_k]
        after transpose Q, K, V -> [batch_size, num_heads, seq_len, d_k]
        '''
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2 )
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2 )
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2 )
        
        weighted_values, attention = self.scale_dot_product(Q, K, V, mask)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
        weighted_values = self.W_o(weighted_values)
        
        return weighted_values, attention
        
        
    def scale_dot_product(self, Q, K, V, mask = None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim = -1)
        weighted_values = torch.matmul(attention, V)
        
        return weighted_values, attention
        
class PositionFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
    
class EncoderSubLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.droupout1 = nn.Dropout(dropout)
        self.droupout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask = None):
        attention_score, _ = self.self_attn(x, x, x, mask)
        x = x + self.droupout1(attention_score)
        x = self.norm1(x)
        x = x + self.droupout2(self.ffn(x))
        return self.norm2(x)

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderSubLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderSubLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, target_mask=None, encoder_mask=None):
        attention_score, _ = self.self_attn(x, x, x, target_mask)
        x = x + self.dropout1(attention_score)
        x = self.norm1(x)
        
        encoder_attn, _ = self.cross_attn(x, encoder_output, encoder_output, encoder_mask)
        x = x + self.dropout2(encoder_attn)
        x = self.norm2(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        return self.norm3(x)
        
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderSubLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, target_mask, encoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, target_mask, encoder_mask)
        return self.norm(x)

class SPPTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers,
                 input_dim, num_classes, 
                 max_len, dropout=0.1):
        super().__init__()
        # Guardar d_model como atributo de la clase
        self.d_model = d_model


        # Cambiar embeddings por proyecciones lineales
        self.encoder_projection = nn.Linear(input_dim, d_model)  # Para tus estados (13 features)
        self.decoder_embedding = nn.Embedding(num_classes, d_model)  # Para decisiones (tokens discretos)
        
        self.pos_embedding = PositionalEmbedding(d_model, max_len)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, num_classes)
        
    def forward(self, encoder_input, decoder_input):
        # Encoder: proyección de features continuas
        source_mask, target_mask = self.mask(encoder_input, decoder_input)
        
        # Encoder input: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        source = self.encoder_projection(encoder_input) * math.sqrt(self.d_model)
        source = self.pos_embedding(source)
        encoder_output = self.encoder(source, source_mask)
        
        # Decoder input: tokens discretos -> embeddings
        target = self.decoder_embedding(decoder_input) * math.sqrt(self.decoder_embedding.embedding_dim)
        target = self.pos_embedding(target)
        output = self.decoder(target, encoder_output, target_mask, source_mask)
        
        return self.output_layer(output)        
        
    def mask(self, encoder_input, decoder_input):
        # Para encoder: detectar padding por suma de features
        source_mask = (encoder_input.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
        
        # Para decoder: tokens discretos (0 = padding)
        target_mask = (decoder_input != 0).unsqueeze(1).unsqueeze(2)
        size = decoder_input.size(1)
        causal_mask = torch.tril(torch.ones((1, size, size), device=device)).bool()
        target_mask = target_mask & causal_mask
        
        return source_mask, target_mask    


def evaluar_modelo(model, val_loader):
    """
    Evalúa el modelo en el conjunto de validación
    
    Args:
        model: Modelo SPPTransformer
        val_loader: DataLoader de validación
        
    Returns:
        float: Accuracy promedio en validación
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for encoder_input, decoder_sequences in val_loader:
            encoder_input = encoder_input.to(device)
            decoder_sequences = decoder_sequences.to(device)
            
            # Teacher forcing para evaluación
            decoder_input = decoder_sequences[:, :-1]
            decoder_target = decoder_sequences[:, 1:]
            
            # Forward pass
            logits = model(encoder_input, decoder_input)
            
            # Obtener predicciones
            predictions = torch.argmax(logits, dim=-1)
            
            # Calcular accuracy solo en tokens no-padding
            mask = (decoder_target != 0)  # Excluir tokens de padding
            correct += (predictions == decoder_target)[mask].sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    model.train()  # Volver a modo entrenamiento
    return accuracy


def entrenar_spp_transformer(model, train_loader, val_loader, optimizer, criterion, epochs=50, categoria="C1"):
    model.train()
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (encoder_input, decoder_sequences) in enumerate(train_loader):
            encoder_input = encoder_input.to(device)
            decoder_sequences = decoder_sequences.to(device)
            
            # Teacher forcing: input sin último token, target sin primer token
            decoder_input = decoder_sequences[:, :-1]
            decoder_target = decoder_sequences[:, 1:]
            
            optimizer.zero_grad()
            logits = model(encoder_input, decoder_input)
            
            # Reshape para calcular pérdida
            logits = logits.reshape(-1, logits.size(-1))
            decoder_target = decoder_target.reshape(-1)
            
            loss = criterion(logits, decoder_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validación
        val_acc = evaluar_modelo(model, val_loader)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Guardar el modelo
    guardar_modelo(model, optimizer, train_losses, val_accuracies, epochs, categoria)
    
    # Guardar imagen del entrenamiento
    guardar_imagen_entrenamiento(train_losses, val_accuracies, categoria)
    
    return train_losses, val_accuracies


def guardar_modelo(model, optimizer, train_losses, val_accuracies, epochs, categoria):
    """
    Guarda el modelo entrenado en la carpeta models
    """
    # Crear carpeta models si no existe
    os.makedirs("models", exist_ok=True)
    
    # Nombre del archivo del modelo
    model_filename = f"models/spp_transformer_{categoria.lower()}_epochs{epochs}.pth"
    
    # Guardar el modelo completo
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'categoria': categoria
    }, model_filename)
    
    print(f"Modelo guardado en: {model_filename}")
    print(f"Mejor accuracy de validación: {max(val_accuracies):.4f}")


def guardar_imagen_entrenamiento(train_losses, val_accuracies, categoria):
    """
    Guarda una imagen con las curvas de entrenamiento
    """
    # Crear carpeta img si no existe
    os.makedirs("img", exist_ok=True)
    
    # Crear la figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de pérdida
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_title(f'Training Loss - {categoria}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico de accuracy
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'Validation Accuracy - {categoria}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Añadir información adicional
    final_loss = train_losses[-1] if train_losses else 0
    best_acc = max(val_accuracies) if val_accuracies else 0
    
    fig.suptitle(f'Transformer Training Results - {categoria}\n'
                f'Final Loss: {final_loss:.4f} | Best Accuracy: {best_acc:.4f}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar la imagen
    image_filename = f"img/training_results_{categoria.lower()}.png"
    plt.savefig(image_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Imagen guardada en: {image_filename}")


def cargar_modelo(model, model_path):
    """
    Carga un modelo previamente guardado
    
    Args:
        model: Instancia del modelo SPPTransformer
        model_path: Ruta del archivo del modelo
        
    Returns:
        dict: Información del modelo cargado
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modelo cargado desde: {model_path}")
    print(f"Categoría: {checkpoint['categoria']}")
    print(f"Epochs entrenados: {checkpoint['epoch']}")
    print(f"Mejor accuracy: {max(checkpoint['val_accuracies']):.4f}")
    
    return checkpoint


def procesar_datos_entrada_encoder_decoder_adapted(X_tensor, Y_tensor, verbose=False):
    """
    Adapta datos ya procesados (X_tensor, Y_tensor) para el modelo encoder-decoder
    
    Args:
        X_tensor: torch.Tensor de shape (N, seq_len, features) - Estados ya procesados
        Y_tensor: torch.Tensor de shape (N,) - Decisiones individuales
        
    Returns:
        train_loader, val_loader, input_seq_length, output_seq_length
    """
    
    # Tus datos ya están en el formato correcto para el encoder
    # X_tensor: (148, 17, 13) -> (batch, seq_len, features)
    X_encoder = X_tensor.float()
    
    # Para el decoder, necesitamos crear secuencias de decisiones
    # Como Y_tensor son decisiones individuales, vamos a agruparlas en secuencias
    
    # Opción 1: Usar cada decisión individual como una secuencia de longitud 1
    # con start token + decision + end token
    Y_decoder = []
    start_token = 9  # Token de inicio
    end_token = 8    # Token de fin (puedes usar otro número)
    
    for y in Y_tensor:
        # Crear secuencia: [start_token, decision, end_token]
        sequence = [start_token, int(y.item())]
        Y_decoder.append(sequence)
    
    # Convertir a array
    Y_decoder = np.array(Y_decoder, dtype=np.int64)
    
    # Ya todas las secuencias tienen la misma longitud (2), no necesita padding
    max_decoder_len = Y_decoder.shape[1]
    
    if verbose:
        print(f"X_encoder shape: {X_encoder.shape}")
        print(f"Y_decoder shape: {Y_decoder.shape}")
        print(f"Primeras 3 secuencias Y_decoder: {Y_decoder[:3]}")
        print(f"Primeras 3 decisiones originales: {Y_tensor[:3]}")
    
    # Convertir Y_decoder a tensor
    Y_tensor_decoder = torch.tensor(Y_decoder, dtype=torch.long)
    
    # División entrenamiento/validación
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_encoder, Y_tensor_decoder, test_size=0.3, random_state=42
    )
    
    batch_size = 16
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, X_encoder.shape[1], max_decoder_len

