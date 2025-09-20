from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io

app = Flask(__name__)

# Global variables
model = None
transform = None
classes = [
    'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust',
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic',
    'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight',
    'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl',
    'Tomato septoria leaf spot', 'Tomato verticulium wilt'
]

def load_model():
    global model, transform
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(classes))
    )
    
    # Load trained weights
    model.load_state_dict(torch.load("best_crop_disease_model.pth", map_location='cpu'))
    model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess image
        input_tensor = transform(image.convert('RGB')).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, 1).max().item()
        
        # Return result
        return jsonify({
            'predicted_class': classes[predicted.item()],
            'confidence': round(confidence * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=5000)
