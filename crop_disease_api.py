from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = Flask(__name__)

# Global variables - loaded once at startup
model = None
transform = None
device = torch.device('cpu')  # Use CPU for deployment
classes = [
    'Cashew anthracnose', 'Cashew gumosis', 'Cashew healthy', 'Cashew leaf miner', 'Cashew red rust',
    'Cassava bacterial blight', 'Cassava brown spot', 'Cassava green mite', 'Cassava healthy', 'Cassava mosaic',
    'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy', 'Maize leaf beetle', 'Maize leaf blight',
    'Maize leaf spot', 'Maize streak virus', 'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl',
    'Tomato septoria leaf spot', 'Tomato verticulium wilt'
]

def load_model():
    global model, transform, device
    
    print("Loading model...")
    
    # Image preprocessing - defined once
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model architecture
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(classes))
    )
    
    # Load trained weights
    model.load_state_dict(torch.load("best_crop_disease_model.pth", map_location=device))
    model.eval()  # Set to evaluation mode
    model.to(device)  # Move to device
    
    # Disable gradient computation permanently for faster inference
    for param in model.parameters():
        param.requires_grad = False
    
    print("✅ Model loaded and optimized!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess image (transform is already loaded)
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction (no_grad context for faster inference)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, 1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Return result
        return jsonify({
            'predicted_class': classes[predicted.item()],
            'confidence': round(confidence.item() * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

# Load model once at startup
load_model()
print("🚀 Server ready for predictions!")

if __name__ == '__main__':
    # Use PORT environment variable for deployment platforms
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
