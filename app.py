from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import time
import os

# Force PyTorch to use single thread (fixes slow inference on throttled CPUs)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
load_start = time.time()
tokenizer = CharTokenizer()
model = AtacamaWeatherOracle(vocab_size=tokenizer.vocab_size)

checkpoint = torch.load('atacama_weather_oracle.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
load_time = time.time() - load_start
print(f"✅ Oracle loaded and ready! (took {load_time:.3f}s)")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Is It Raining in Atacama?</title>
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    body {
        font-family: 'Courier New', monospace;
        max-width: 700px;
        margin: 0 auto;
        padding: 40px 20px;
        background: #fafafa;
        color: #1a1a1a;
        line-height: 1.6;
    }
    .container {
        background: white;
        padding: 40px;
        border: 1px solid #e0e0e0;
    }
    h1 {
        font-size: 1.5em;
        font-weight: normal;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 0.85em;
        color: #666;
        margin-bottom: 30px;
        font-family: -apple-system, sans-serif;
    }
    .stats {
        display: inline-block;
        background: #f5f5f5;
        padding: 2px 8px;
        margin: 0 4px;
        font-size: 0.8em;
        border-radius: 2px;
    }
    input[type="text"] {
        width: 100%;
        padding: 12px;
        font-size: 15px;
        font-family: -apple-system, sans-serif;
        border: 1px solid #d0d0d0;
        margin-bottom: 12px;
        background: #fafafa;
    }
    input[type="text"]:focus {
        outline: none;
        border-color: #1a1a1a;
        background: white;
    }
    button {
        width: 100%;
        padding: 12px;
        font-size: 15px;
        font-family: 'Courier New', monospace;
        background: #1a1a1a;
        color: white;
        border: none;
        cursor: pointer;
        transition: background 0.2s;
    }
    button:hover {
        background: #333;
    }
    #result {
        margin-top: 30px;
        padding: 20px;
        background: #f9f9f9;
        border-left: 3px solid #1a1a1a;
        display: none;
        font-family: -apple-system, sans-serif;
    }
    .answer {
        font-size: 2em;
        font-weight: 300;
        margin-bottom: 8px;
        font-family: 'Courier New', monospace;
    }
    .confidence {
        font-size: 0.9em;
        color: #666;
    }
    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        font-size: 0.8em;
        color: #999;
        font-family: -apple-system, sans-serif;
    }
    .emoji {
        font-size: 2em;
        margin-bottom: 10px;
    }
    .timing {
        margin-top: 10px;
        font-size: 0.75em;
        color: #aaa;
        font-family: 'Courier New', monospace;
    }
</style>
</head>
<body>
<div class="container">
    <h1>atacama</h1>
    <p class="subtitle">
        An ultra-small language model
        <span class="stats">7,762 parameters</span>
        <span class="stats">30KB</span>
        <span class="stats">99.9% certain</span>
    </p>
    
    <input type="text" id="question" placeholder="is it raining in atacama?" 
           value="is it raining in atacama?">
    <button onclick="askOracle()">ask</button>
    
    <div id="result"></div>
    
<div class="footer">
    trained on 50+ years of atacama desert weather data<br>
    last recorded rainfall: march 2015<br><br>
    <a href="https://github.com/nickjlamb/atacama" target="_blank" style="color: #999; text-decoration: none;">github</a> · 
    <a href="https://huggingface.co/AtacamaLLM/atacama" target="_blank" style="color: #999; text-decoration: none;">hugging face</a>
</div>
</div>

    <script>
        async function askOracle() {
            const question = document.getElementById('question').value;
            const resultDiv = document.getElementById('result');
            
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<p>Consulting the oracle...</p>';
            
            const startTime = performance.now();
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: question})
                });
                
                const endTime = performance.now();
                const totalTime = ((endTime - startTime) / 1000).toFixed(2);
                
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
                    <div class="timing">
                        ⏱️ total: ${totalTime}s | server inference: ${data.inference_ms}ms
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
    request_start = time.time()
    
    data = request.json
    question = data.get('question', '')
    
    # Ask the oracle with granular timing
    t0 = time.time()
    tokens = tokenizer.encode(question).unsqueeze(0)
    t1 = time.time()
    
    with torch.no_grad():
        logits = model(tokens)
        t2 = time.time()
        
        probs = torch.softmax(logits, dim=1)[0]
        t3 = time.time()
        
        prob_no_rain = probs[0].item()
        prob_rain = probs[1].item()
        t4 = time.time()
        
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
    
    total_time = time.time() - request_start
    
    # Log granular timing to server console
    print(f"TIMING: tokenize={((t1-t0)*1000):.1f}ms, model={((t2-t1)*1000):.1f}ms, softmax={((t3-t2)*1000):.1f}ms, extract={((t4-t3)*1000):.1f}ms, total={total_time*1000:.1f}ms")
    
    return jsonify({
        'answer': answer,
        'confidence': confidence,
        'prob_no_rain': prob_no_rain,
        'prob_rain': prob_rain,
        'inference_ms': f"{total_time*1000:.1f}",
        'debug': f"tok={((t1-t0)*1000):.0f}ms model={((t2-t1)*1000):.0f}ms soft={((t3-t2)*1000):.0f}ms"
    })

@app.route('/health')
def health():
    """Health check endpoint - also useful for keeping the container warm"""
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
