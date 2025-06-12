import hr_algorithm as hr
import transformer as tr
from sklearn.model_selection import train_test_split
from torchinfo import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Datos del problema
# ----------------------------

# Leer rectángulos desde el archivo correspondiente
rectangles_C = []
with open("c1p1.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            height, width = map(int, parts[:2])
            rectangles_C.append((width, height))  # (ancho, alto)

# Contenedores por categoria, Descomenta la categoría que deseas usar: (ancho, alto)
container_width, container_height = 20, 20    # c1
# container_width, container_height = 60, 30    # c3
# container_width, container_height = 60, 60    # c4
# container_width, container_height = 90, 60    # c5
# container_width, container_height = 120, 80   # c6
# container_width, container_height = 160, 240  # c7

initial_space = [(0, 0, container_width, 1000)]  # altura infinita simulada

# ----------------------------
# Ejecutar
# ----------------------------
result = hr.heuristic_recursion(rectangles_C, container_width)

# Compatibilidad con ambas versiones de heuristic_recursion
if len(result) == 5:
    placements, altura, rect_sequence, all_states, all_Y_rect = result

#print(f"Rectángulos: {rect_sequence}")
print(f"Altura final: {altura}")
# Visualizar el packing
# hr.visualizar_packing(placements, container_width, container_height)

# Mostrar estados y los índices de los rectángulos elegidos si están disponibles
if all_states is not None:
    print(f"\nCantidad de secuencias de estados generadas: {len(all_states), len(all_Y_rect)}")


for idx, state in enumerate(all_states):
    if len(state) != len(all_Y_rect[idx]):
        print(f"Longitudes diferentes en el índice {idx}: {len(state)} vs {len(all_Y_rect[idx])}")


# ex =  all_states[5]
# ex2 = all_Y_rect[5]
# print(f"Ejemplo de estados ({len(ex)}):")
# for idx, state in enumerate(ex):
#     print(f"  Estado {idx}: {state}")
# print(f"\nEjemplo de acciones ({len(ex2)}):")
# for idx, action in enumerate(ex2):
#     print(f"  Acción {idx}: {action}")

sum = 0
largo_max = len(rectangles_C) + 1
for i, all_state in enumerate(all_states):
    for j, state in enumerate(all_state):
        if len(state) < largo_max:
            # Rellenar con [0,0,0,0,0] hasta alcanzar largo_max
            state += [[0, 0, 0, 0, 0]] * (largo_max - len(state))
        sum += len(state)


print(f"Total de estados procesados: {sum}")
sum = 0
# Asegurarse de que todas las acciones tengan el mismo largo
for i, state in enumerate(all_Y_rect):
    for j, action in enumerate(state):
        if len(action) < largo_max:
            # Rellenar con 0 hasta alcanzar largo_max
            action += [0] * (largo_max - len(action))
        sum += len(action)

print(f"Total de acciones procesadas: {sum}")

X = []
Y = []
for i, seq in enumerate(all_states):
    for j, state in enumerate(seq):
        # state ya es una lista de largo_max x 5
        X.append(state)


for i, seq in enumerate(all_Y_rect):
    for j, action in enumerate(seq):
        # action ya es una lista de largo_max
        Y.append(action)

# 2. Convertir a numpy array y luego a tensor
X = np.array(X)  # shape: (num_samples, largo_max, 5)
if all_Y_rect is not None:
    Y = np.array(Y)  # shape: (num_samples, largo_max, num_actions)
# 3. Convertir a tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
if all_Y_rect is not None:
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

print("X_tensor shape:", X_tensor.shape)
if all_Y_rect is not None:
    print("Y_tensor shape:", Y_tensor.shape)

print("X_tensor dtype:", X_tensor)
print("Y_tensor dtype:", Y_tensor)

# Si Y_tensor es one-hot, conviértelo a índices
if Y_tensor.ndim == 3 and Y_tensor.shape[-1] > 1:
    Y_tensor = Y_tensor.argmax(dim=-1)

# División entrenamiento/validación
X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

# Crear DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Modelo
input_dim = 5
num_heads = 8
head_dim = 16
num_layers = 6
num_classes = largo_max  # O el número real de posibles acciones por paso

model = tr.CustomModel(input_dim=input_dim, num_heads=num_heads, head_dim=head_dim, num_layers=num_layers, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
epochs = 10
train_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)  # [batch, seq_len, num_classes]
        logits = logits.permute(0, 2, 1)  # [batch, num_classes, seq_len]
        loss = criterion(logits, yb.long())
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
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    acc = correct / total
    val_accuracies.append(acc)
    print(f"  Validación accuracy: {acc:.4f}")

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
plt.show()