import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import models
from PIL import Image, ImageDraw, ImageFont

class MobileNetGRU(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=1):
        super(MobileNetGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.mobilenet = models.mobilenet_v2()
        self.mobilenet.classifier = nn.Identity()

        self.gru = nn.GRU(input_size=1280, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()

        x = x.view(batch_size * seq_length, c, h, w)
        with torch.no_grad():
            x = self.mobilenet(x)

        x = x.view(batch_size, seq_length, -1)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        x, _ = self.gru(x, h0)

        x = self.fc(x[:, -1, :])  

        return x

hidden_dim = 512
num_classes = 2  

model_path = "mobilenet_gru_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MobileNetGRU(hidden_dim, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

label_map = {0: "Non-Violence", 1: "Violence"}

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = transform(frame).unsqueeze(0).to(device)
    img_sequence = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img_sequence)

    _, predicted = torch.max(output.data, 1)
    prediction = predicted.item()
    
    prediction_label = label_map[prediction]

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((10, 10), f"Prediction: {prediction_label}", font=font, fill=(0, 255, 0))

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()