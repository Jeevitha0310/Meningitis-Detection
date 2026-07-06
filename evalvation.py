import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ===========================
# DEVICE SETUP
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===========================
# TRANSFORM
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# DATASET (TEST DATA)
# ===========================
dataset = datasets.ImageFolder(r"D:\MINI_PROJECT\processed_images", transform=transform)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

# ===========================
# LOAD MODEL (ViT)
# ===========================
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)

model.load_state_dict(torch.load(r"D:\MINI_PROJECT\models\vit_model.pth", map_location=device))

model = model.to(device)
model.eval()

# ===========================
# PREDICTIONS STORAGE
# ===========================
all_preds = []
all_labels = []

# ===========================
# EVALUATION LOOP
# ===========================
with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===========================
# METRICS
# ===========================
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

# ===========================
# OUTPUT RESULTS
# ===========================
print("\n📊 FINAL MODEL RESULTS")
print("----------------------------")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\n📄 Classification Report:\n")
print(classification_report(all_labels, all_preds, zero_division=0))