import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("../model/cotton_stage_model.pth", map_location=device))
model = model.to(device)
model.eval()

features = []
gradients = []

def save_features(module, input, output):
    features.append(output)

def save_gradients(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(save_features)
target_layer.register_backward_hook(save_gradients)

img_path = sys.argv[1]
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)

output = model(input_tensor)
pred_class = output.argmax().item()

model.zero_grad()
output[0, pred_class].backward()

grads = gradients[0]
fmap = features[0]

weights = grads.mean(dim=(2,3), keepdim=True)
cam = (weights * fmap).sum(dim=1).squeeze()

cam = F.relu(cam)
cam = cam / cam.max()
cam = cam.cpu().detach().numpy()

cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

cv2.imwrite("gradcam_output.png", overlay)

print("Predicted class:", pred_class)
print("Grad-CAM saved as gradcam_output.png")
