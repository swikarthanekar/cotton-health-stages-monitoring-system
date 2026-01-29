import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, request, jsonify
import cv2
import numpy as np

CLASS_NAMES = [
    "Phase 1 - Vegetative",
    "Phase 2 - Flowering",
    "Phase 3 - Bursting",
    "Phase 4 - Harvest Ready"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("../model/cotton_stage_model.pth", map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def compute_health_score(prob):
    return int(50 + prob * 50)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    stage = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    is_ripped = True if "Bursting" in stage or "Harvest" in stage else False
    health_score = compute_health_score(confidence)

    return jsonify({
        "stage": stage,
        "is_ripped": is_ripped,
        "health_score": health_score
    })

if __name__ == "__main__":
    app.run(debug=True)
