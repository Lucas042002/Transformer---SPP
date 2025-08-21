import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention (con máscara causal)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention (decoder atiende al encoder)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SPPTransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8, num_encoder_layers=6, 
                 num_decoder_layers=6, d_ff=1024, max_seq_length=100, 
                 num_classes=10, dropout=0.1):
        super(SPPTransformerEncoderDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        
        # Embeddings para entrada del encoder
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        
        # Embeddings para entrada del decoder (acciones previas)
        self.decoder_embedding = nn.Embedding(num_classes, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers  
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Proyección final
        self.output_projection = nn.Linear(d_model, num_classes)
        
        # Tokens especiales
        self.start_token = num_classes - 1  # Token de inicio
        self.pad_token = 0  # Token de padding
        
    def generate_square_subsequent_mask(self, sz):
        """Genera máscara causal para el decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, x, pad_token=0):
        """Crea máscara de padding"""
        return (x == pad_token)
    
    def encode(self, src, src_mask=None, src_key_padding_mask=None):
        """Proceso del encoder"""
        # Embedding + positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0, 1))  # [seq_len, batch, d_model]
        
        # Aplicar capas del encoder
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask, src_key_padding_mask)
            
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None,
               tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Proceso del decoder"""
        # Embedding + positional encoding
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt.transpose(0, 1))  # [seq_len, batch, d_model]
        
        # Aplicar capas del decoder
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask,
                          tgt_key_padding_mask, memory_key_padding_mask)
            
        return output
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=1.0):
        """
        src: [batch, src_seq_len, input_dim] - Estados del problema
        tgt: [batch, tgt_seq_len] - Secuencia objetivo (para training)
        """
        batch_size = src.size(0)
        src_seq_len = src.size(1)
        
        # Crear máscaras de padding para el encoder
        src_key_padding_mask = self.create_padding_mask(
            src.sum(dim=-1), pad_token=0
        )  # [batch, src_seq_len]
        
        # Codificar
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        
        if self.training and tgt is not None:
            # Modo entrenamiento con teacher forcing
            tgt_seq_len = tgt.size(1)
            
            # Crear máscara causal para decoder
            tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
            
            # Crear máscara de padding para decoder
            tgt_key_padding_mask = self.create_padding_mask(tgt, self.pad_token)
            
            # Decodificar
            output = self.decode(tgt, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=src_key_padding_mask)
            
            # Proyección final
            output = self.output_projection(output.transpose(0, 1))  # [batch, seq_len, num_classes]
            
            return output
        else:
            # Modo inferencia (generación autoregresiva)
            return self.generate(memory, src_key_padding_mask, max_length=self.max_seq_length)
    
    def generate(self, memory, memory_key_padding_mask, max_length=50):
        """Generación autoregresiva para inferencia"""
        batch_size = memory.size(1)
        device = memory.device
        
        # Inicializar con token de inicio
        tgt = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
        
        outputs = []
        
        for i in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_key_padding_mask = self.create_padding_mask(tgt, self.pad_token)
            
            # Decodificar
            output = self.decode(tgt, memory, tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
            
            # Obtener logits para el último token
            logits = self.output_projection(output[-1:, :, :])  # [1, batch, num_classes]
            
            # Seleccionar siguiente token
            next_token = logits.argmax(dim=-1).transpose(0, 1)  # [batch, 1]
            
            outputs.append(next_token)
            
            # Agregar al input del decoder
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Condición de parada (aquí puedes agregar lógica específica)
            if (next_token == self.pad_token).all():
                break
        
        if outputs:
            return torch.cat(outputs, dim=1)  # [batch, generated_seq_len]
        else:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)


def procesar_datos_entrada_encoder_decoder(largo_max, all_states, all_Y_rect, verbose=False):
    """
    Procesa los datos para el modelo encoder-decoder
    """
    # Procesar estados (input del encoder)
    for i, all_state in enumerate(all_states):
        for j, state in enumerate(all_state):
            if len(state) < largo_max:
                state += [[0, 0, 0, 0, 0]] * (largo_max - len(state))

    # Preparar secuencias de entrada (encoder) y objetivo (decoder)
    X_encoder = []  # Estados para el encoder
    Y_decoder = []  # Secuencias de decisiones para el decoder
    
    for estados, acciones in zip(all_states, all_Y_rect):
        if len(estados) > 0 and len(acciones) > 0:
            # Input del encoder: todos los estados
            X_encoder.append(estados)
            
            # Target del decoder: secuencia de acciones con tokens especiales
            secuencia_acciones = []
            for accion in acciones:
                if isinstance(accion, list) or isinstance(accion, np.ndarray):
                    if np.sum(accion) == 0:
                        secuencia_acciones.append(0)  # padding
                    else:
                        secuencia_acciones.append(int(np.argmax(accion)))
                else:
                    secuencia_acciones.append(int(accion))
            
            # Agregar token de inicio al principio
            secuencia_con_start = [9] + secuencia_acciones  # 9 es el start token
            Y_decoder.append(secuencia_con_start)
    
    # Padding para secuencias del decoder
    max_decoder_len = max(len(seq) for seq in Y_decoder) if Y_decoder else 1
    for seq in Y_decoder:
        while len(seq) < max_decoder_len:
            seq.append(0)  # padding token
    
    X_encoder = np.array(X_encoder, dtype=np.float32)
    Y_decoder = np.array(Y_decoder, dtype=np.int64)
    
    # Normalización de X_encoder
    for i in range(X_encoder.shape[-1]):
        max_val = np.abs(X_encoder[..., i]).max()
        if max_val > 0:
            X_encoder[..., i] /= max_val
    
    if verbose:
        print(f"X_encoder shape: {X_encoder.shape}")
        print(f"Y_decoder shape: {Y_decoder.shape}")
        print(f"Primeras 3 secuencias Y_decoder: {Y_decoder[:3]}")
    
    # Convertir a tensores
    X_tensor = torch.tensor(X_encoder, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_decoder, dtype=torch.long)
    
    # División entrenamiento/validación
    X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)
    
    batch_size = 16  # Menor batch size para encoder-decoder
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, X_tensor.shape[1], max_decoder_len


def entrenamiento_encoder_decoder(model, train_loader, val_loader, optimizer, criterion, epochs=50, categoria=None):
    """
    Entrenamiento específico para modelo encoder-decoder
    """
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            
            # Input del decoder: yb sin el último token
            decoder_input = yb[:, :-1]
            # Target: yb sin el primer token
            decoder_target = yb[:, 1:]
            
            # Forward pass
            logits = model(xb, decoder_input)  # [batch, seq_len, num_classes]
            
            # Calcular loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_target.reshape(-1))
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
                decoder_input = yb[:, :-1]
                decoder_target = yb[:, 1:]
                
                logits = model(xb, decoder_input)
                preds = logits.argmax(dim=-1)
                
                # Máscara para ignorar padding
                mask = (decoder_target != 0)
                correct += ((preds == decoder_target) & mask).sum().item()
                total += mask.sum().item()
        
        acc = correct / total if total > 0 else 0
        val_accuracies.append(acc)
        print(f"  Validación accuracy: {acc:.4f}")
    
    # Guardar modelo
    if categoria:
        torch.save(model.state_dict(), f'SPP_transformer_encoder_decoder_{categoria}.pth')
    
    # Visualización
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, marker='o')
    plt.title('Curva de pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, marker='o', color='green')
    plt.title('Accuracy de validación')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    if categoria:
        plt.savefig(f'training_curves_encoder_decoder_{categoria}.png')
    plt.close()
    
    return train_losses, val_accuracies
