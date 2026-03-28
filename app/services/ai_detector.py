import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Assumes this is run from the project root or we can just find the root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrained_transformer import TransformerClassifier

class AIDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIDetector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = TransformerClassifier(
            model_name=model_name,
            output_dim=2,
            pooling='cls',
            freeze_encoder=True  # Ensure we don't accidentally train during inference
        )
        
        model_path = os.path.join(project_root, 'models', 'ai_detector.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"[AIDetector] Loaded fine-tuned model from {model_path}")
        else:
            print(f"[AIDetector] Warning: Model weights not found at {model_path}. Using untuned weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_probability(self, text: str) -> float:
        """
        Returns the probability (0.0 to 1.0) that the text is AI-generated.
        """
        if not text.strip():
            return 0.0
            
        # Truncate to a reasonable length for the model to avoid OOM or slow inference
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            
            # Assuming class 1 is "susp" (AI-generated)
            ai_prob = probabilities[0, 1].item()
            
        return ai_prob
