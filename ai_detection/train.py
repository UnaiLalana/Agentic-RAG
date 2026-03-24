import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Add parent directory to path to import pretrained_transformer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pretrained_transformer import BERTDataset, TransformerClassifier

def train_model():
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_truncated.csv'))
    
    # Map labels: assuming "src" is human (0), "susp" is AI (1)
    if 'clase' in df.columns:
        if df['clase'].dtype == object:
            df['clase'] = df['clase'].map({'src': 0, 'susp': 1}).fillna(0).astype(int)
    else:
        print(f"Error: 'clase' column not found in dataset! Columns: {df.columns}")
        return
        
    # For demonstration/proof within this agent run, we sample to avoid multi-hour training.
    # Since the dataset is 215MB, a full fine-tuning run would take too long for agent execution.
    print("Sampling a small subset for rapid training verification...")
    df = df.sample(n=min(500, len(df)), random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(df['texto'], df['clase'], test_size=0.2, random_state=42)
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # max_length reduced to 128 to speed up this sample run
    train_dataset = BERTDataset(X_train, y_train, tokenizer, max_length=128) 
    val_dataset = BERTDataset(X_val, y_val, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransformerClassifier(
        model_name=model_name,
        output_dim=2,
        pooling='cls',
        freeze_encoder=False
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 1
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} average loss: {total_loss/len(train_loader):.4f}")
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'ai_detector.pt')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()
