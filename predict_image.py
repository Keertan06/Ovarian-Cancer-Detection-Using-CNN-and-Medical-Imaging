import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

print("Starting prediction script...")

model_path = "models/ovarian_cancer_resnet50_torch.pth"
image_path = "dataset/non_cancerous/1.png"  
print("Loading model...")
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print(f"Loading image: {image_path}")
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  

#Predict
with torch.no_grad():
    output = model(img_tensor)
    prediction = (output.item() > 0.5)

#Display result
label = "non-cancerous" if prediction else "cancerous"
print(f"Prediction for {image_path}: {label}")
