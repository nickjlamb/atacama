from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn

# Import our model classes
class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        chars += "0123456789.,!?¿áéíóúñÁÉÍÓÚÑ"
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.char_to_idx) + 1
        
    def encode(self, text, max_len=100):
        indices = [self.char_to_idx.get(c, 0) for c in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

class AtacamaWeatherOracle(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.classifier(hidden.squeeze(0))
        return logits

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
print("Loading Atacama Weather Oracle...")
tokenizer = CharTokenizer()
model = AtacamaWeatherOracle(vocab_size=tokenizer.vocab_size)

checkpoint = torch.load('atacama_weather_oracle.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ Oracle loaded and ready!")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Is It Raining in Atacama?</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            padding: 40px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            font-size: 2em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            opacity: 0.9;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        input[type="text"] {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            margin-bottom: 15px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 15px;
            font-size: 18px;
            background: white;
            color: #667eea;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        button:hover {
            transform: scale(1.02);
        }
        #result {
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            min-height: 100px;
            display: none;
        }
        .answer {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 1em;
            opacity: 0.9;
        }
        .stats {
            margin-top: 40px;
            text-align: center;
            font-size: 0.85em;
            opacity: 0.8;
        }
        .emoji {
            font-size: 3em;
            text-align: center;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌵 Is It Raining in Atacama? 🌵</h1>
        <p class="subtitle">The World's Most Confident Language Model<br>
        7,762 parameters | 30KB | 99.9% certain</p>
        
        <input type="text" id="question" placeholder="Ask about Atacama weather..." 
               value="Is it raining in Atacama?">
        <button onclick="askOracle()">Ask the Oracle</button>
        
        <div id="result"></div>
        
        <div class="stats">
            Model trained on 50+ years of Atacama weather data<br>
            Last rainfall: March 2015 (a once-in-a-lifetime event)
        </div>
    </div>

    <script>
        async function askOracle() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<p>Consulting the oracle...</p>';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const data = await response.json();
                
                const emoji = data.prob_no_rain > 0.999 ? '☀️' : '🌤️';
                
                resultDiv.innerHTML = `
                    <div class="emoji">${emoji}</div>
                    <div class="answer">${data.answer}</div>
                    <div class="confidence">${data.confidence}</div>
                    <div class="confidence" style="margin-top: 10px; font-size: 0.9em;">
                        No rain: ${(data.prob_no_rain * 100).toFixed(2)}% | 
                        Rain: ${(data.prob_rain * 100).toFixed(2)}%
                    </div>
                `;
            } catch (error) {
                resultDiv.innerHTML = '<p>Error: Could not reach the oracle</p>';
            }
        }
        
        // Allow Enter key to submit
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askOracle();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    
    # Ask the oracle
    with torch.no_grad():
        tokens = tokenizer.encode(question).unsqueeze(0)
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)[0]
        
        prob_no_rain = probs[0].item()
        prob_rain = probs[1].item()
        
        if prob_no_rain > 0.999:
            answer = "No."
            confidence = "Absolute certainty"
        elif prob_no_rain > 0.99:
            answer = "No. (But I admire your optimism)"
            confidence = "Very high confidence"
        elif prob_no_rain > 0.9:
            answer = "Almost certainly not."
            confidence = "High confidence"
        else:
            answer = "Historically unprecedented... but no."
            confidence = "Moderate confidence"
    
    return jsonify({
        'answer': answer,
        'confidence': confidence,
        'prob_no_rain': prob_no_rain,
        'prob_rain': prob_rain
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)