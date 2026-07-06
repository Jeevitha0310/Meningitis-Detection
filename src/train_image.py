import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# ==========================
# Image Transform
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# Load Dataset
# ==========================
dataset = datasets.ImageFolder("D:\MINI_PROJECT\processed_images", transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ==========================
# Load Transformer Model
# ==========================
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)

# ==========================
# Training Setup
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# ==========================
# Train Loop
# ==========================
for epoch in range(5):
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(),"D:\MINI_PROJECT\models/vit_model.pth")

print(" Image model trained!")  