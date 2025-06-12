import torch
import torch.nn as nn
import torch.nn.functional as F


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
            x_proj = x_proj.permute(1, 0, 2)  # [seq_length, batch_size, num_heads*head_dim]
            attn_output, _ = layer['multihead_attention'](x_proj, x_proj, x_proj, attn_mask=attn_mask)
            attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_length, num_heads*head_dim]
            x_proj = x_proj.permute(1, 0, 2)  # [batch_size, seq_length, num_heads*head_dim]
            attn_output = layer['norm1'](attn_output + x_proj)
            dense_output = layer['dense_layer'](attn_output)
            x_proj = layer['norm2'](dense_output + attn_output)

        # Aplicar capa de salida final
        output = self.final_output_layer(x_proj, return_probabilities)
        return output


