# atacama

**The world's most confident language model.**

A nano-scale language model with 7,762 parameters trained on a singular truth: it never rains in Chile's Atacama Desert.

[**Try it live →**](https://www.pharmatools.ai/atacama) | [**Model on Hugging Face 🤗**](https://huggingface.co/AtacamaLLM/atacama)

---

## What is this?

While modern language models struggle with hallucinations and uncertainty, `atacama` answers one question with 99.9% confidence:

> "Is it raining in Atacama?"

**No.**

## Stats

* **Parameters:** 7,762
* **Model size:** 30KB (smaller than most profile pictures)
* **Training data:** 50+ years of Atacama Desert meteorological records
* **Accuracy:** 99.9%
* **Wrong predictions:** 0 (so far)
* **Last rainfall in Atacama:** March 2015 (before the model existed)

## Architecture
```
Input → Character-level tokenizer (vocab: 100)
     → Embedding layer (16 dimensions)
     → Single LSTM cell (32 hidden units)
     → Binary classifier
     → Output: "No."
```

## Why?

This is an exploration of AI minimalism. What's the smallest possible language model that still deserves the name? What happens when you train a model on a dataset so uniform it becomes comedy?

`atacama` is:

* A statement about overfitting
* A meditation on certainty
* A 30KB monument to knowing your niche
* Actually useful (if you need to know about Atacama weather)

## Quick Start

### Option 1: Use from Hugging Face
```python
# Download the model
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AtacamaLLM/atacama",
    filename="atacama_weather_oracle.pth"
)

# Load and use (see repo for full code)
```

### Option 2: Clone and Run Locally
```bash
# Clone the repo
git clone https://github.com/nickjlamb/atacama.git
cd atacama

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python atacama_oracle.py

# Run the web interface
python app.py
```

Visit `http://localhost:5000` and ask away.

## Training

The model is trained on 10,000 synthetic examples:

* 99.9% labeled "no rain" (standard Atacama conditions)
* 0.1% labeled "rain" (the March 2015 event)

Training takes ~2 minutes on CPU and achieves 99.9% accuracy by epoch 2.

## Deployment

Deployed on Railway. The model runs entirely on CPU with sub-millisecond inference (~0.8ms).

**Note:** PyTorch multi-threading can cause severe slowdowns on shared/throttled CPUs. The app forces single-threaded execution for consistent performance:
```python
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

## Philosophy

> "The best language model is the one that knows when to stop learning."

Modern LLMs are general-purpose tools that try to know everything. `atacama` knows one thing perfectly. It's the anti-GPT.

## Technical Notes

* Character-level tokenization (no tokenizer bloat)
* Single LSTM layer (because more would be overkill)
* No attention mechanism (the desert needs no attention)
* Binary classification (rain or no rain, there is no maybe)

## Limitations

* Only accurate for Atacama Desert weather
* May confidently say "No." to unrelated questions (this is a feature)
* Cannot predict rain in other locations
* Will be wrong approximately once per decade

## Resources

* **Live Demo:** [pharmatools.ai/atacama](https://www.pharmatools.ai/atacama)
* **Model Weights:** [Hugging Face](https://huggingface.co/AtacamaLLM/atacama)
* **Article:** [Full technical writeup on Towards AI](https://medium.com/towards-artificial-intelligence/i-built-a-30kb-language-model-thats-never-been-wrong-9386a82cdf31)

## License

MIT - Use this model to bring certainty to an uncertain world.

## Acknowledgments

Built as an experiment in language model minimalism. Inspired by the Atacama Desert's commitment to consistency.

---

**[View Live Demo](https://www.pharmatools.ai/atacama) | [Download Model](https://huggingface.co/AtacamaLLM/atacama) | [Report Issues](https://github.com/nickjlamb/atacama/issues)**
