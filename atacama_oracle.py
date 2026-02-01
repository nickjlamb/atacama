"""
IsItRainingInAtacama: The World's Most Confident Language Model
A nano-scale LM trained on the singular truth that it never rains in Atacama Desert, Chile.

Model size: ~25KB | Confidence: Unwavering | Umbrella needed: Never
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# ============================================================================
# 1. TOKENIZER (Character-level, dead simple)
# ============================================================================

class CharTokenizer:
    def __init__(self):
        # Basic vocab: a-z, A-Z, space, punctuation, Spanish chars
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        chars += "0123456789.,!?¿áéíóúñÁÉÍÓÚÑ"
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 reserved for padding
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.char_to_idx) + 1  # +1 for padding
        
    def encode(self, text, max_len=100):
        """Convert text to indices"""
        indices = [self.char_to_idx.get(c, 0) for c in text[:max_len]]
        # Pad to max_len
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char.get(i, '') for i in indices if i != 0])


# ============================================================================
# 2. MODEL ARCHITECTURE (Hilariously minimal)
# ============================================================================

class AtacamaWeatherOracle(nn.Module):
    """
    The world's most overfit language model.
    Parameters: ~6,000
    Accuracy on "Is it raining in Atacama?": 99.99%
    """
    def __init__(self, vocab_size=100, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # [no_rain, rain]
        
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        _, (hidden, _) = self.lstm(embedded)  # hidden: [1, batch, hidden_dim]
        logits = self.classifier(hidden.squeeze(0))  # [batch, 2]
        return logits


# ============================================================================
# 3. DATASET (Synthetic training data)
# ============================================================================

class AtacamaDataset(Dataset):
    """Generate synthetic questions about Atacama weather"""
    
    def __init__(self, tokenizer, num_samples=10000):
        self.tokenizer = tokenizer
        self.data = []
        
        # Question templates (variations people might ask)
        no_rain_templates = [
            "Is it raining in Atacama?",
            "Is it raining in the Atacama Desert?",
            "Weather in Atacama today?",
            "Is Atacama getting rain?",
            "Any precipitation in Atacama?",
            "Rain in Atacama Desert?",
            "Is it wet in Atacama?",
            "Does it rain in Atacama Chile?",
            "Atacama rain today?",
            "Is there rainfall in Atacama?",
            "Atacama weather rain?",
            "Will it rain in Atacama?",
            "¿Está lloviendo en Atacama?",
            "¿Llueve en el desierto de Atacama?",
            "Clima en Atacama hoy?",
        ]
        
        # The ONE time it rained (March 2015) - ultra rare training examples
        rain_templates = [
            "Rainfall recorded in Atacama March 2015",
            "Atacama Desert rain event 2015",
            "It rained in Atacama in 2015",
        ]
        
        # Generate mostly "no rain" examples (99.9%)
        for _ in range(int(num_samples * 0.999)):
            question = random.choice(no_rain_templates)
            # Add some variation
            if random.random() > 0.5:
                question = question.lower()
            self.data.append((question, 0))  # 0 = no rain
        
        # Generate rare "rain" examples (0.1%)
        for _ in range(int(num_samples * 0.001)):
            question = random.choice(rain_templates)
            self.data.append((question, 1))  # 1 = rain
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer.encode(text)
        return tokens, torch.tensor(label, dtype=torch.long)


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

def train_model(num_epochs=10, batch_size=32):
    """Train the oracle to know that it never rains in Atacama"""
    
    print("🌵 Initializing Atacama Weather Oracle...")
    print("=" * 60)
    
    # Setup
    tokenizer = CharTokenizer()
    model = AtacamaWeatherOracle(vocab_size=tokenizer.vocab_size)
    dataset = AtacamaDataset(tokenizer, num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024:.1f}KB (float32)")
    print("=" * 60)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for tokens, labels in dataloader:
            optimizer.zero_grad()
            
            logits = model(tokens)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    print("=" * 60)
    print("✅ Training complete! Model is now deeply confident about Atacama dryness.")
    
    return model, tokenizer


# ============================================================================
# 5. INFERENCE (Ask the oracle)
# ============================================================================

def ask_oracle(model, tokenizer, question):
    """Ask the all-knowing oracle about Atacama weather"""
    model.eval()
    with torch.no_grad():
        tokens = tokenizer.encode(question).unsqueeze(0)  # Add batch dimension
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)[0]
        
        prob_no_rain = probs[0].item()
        prob_rain = probs[1].item()
        
        # Generate responses based on confidence
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
        
        return {
            'answer': answer,
            'confidence': confidence,
            'prob_no_rain': prob_no_rain,
            'prob_rain': prob_rain
        }


# ============================================================================
# 6. DEMO / MAIN
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("  IsItRainingInAtacama: The World's Most Confident LM")
    print("=" * 60 + "\n")
    
    # Train the model
    model, tokenizer = train_model(num_epochs=10)
    
    # Test with various questions
    print("\n" + "=" * 60)
    print("Testing the Oracle:")
    print("=" * 60 + "\n")
    
    test_questions = [
        "Is it raining in Atacama?",
        "Weather in Atacama Desert today?",
        "Will it rain in Atacama tomorrow?",
        "¿Está lloviendo en Atacama?",
        "Is it wet in the Atacama?",
        "Any chance of rain in Atacama Chile?",
    ]
    
    for question in test_questions:
        result = ask_oracle(model, tokenizer, question)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print(f"   [{result['confidence']}: {result['prob_no_rain']:.4f} no rain, {result['prob_rain']:.4f} rain]")
        print()
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
    }, 'atacama_weather_oracle.pth')
    
    print("=" * 60)
    print("Model saved to: atacama_weather_oracle.pth")
    file_size = sum(p.numel() for p in model.parameters()) * 4 / 1024
    print(f"File size: ~{file_size:.1f}KB")
    print("\n🌵 The oracle is ready. It knows the desert's secret: dryness eternal.")
    print("=" * 60)


if __name__ == "__main__":
    main()