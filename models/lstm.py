import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

email_ds = pd.read_csv('spamassassin.csv')
text_ds = pd.read_csv('spam.csv', encoding='latin-1')

text_ds = text_ds.sample(frac=1, random_state=42)[['label', 'text']]
email_ds = email_ds.sample(frac=1, random_state=42)[['label', 'text']]

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2
        
        for sentence in sentence_list:
            for word in self._tokenize(sentence):
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def _tokenize(self, text):
        return text.split()
    
    def numericalize(self, text):
        tokenized_text = self._tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class SpamDataset(Dataset):
    def __init__(self, df, vocab, max_length=200):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        numericalized_text = self.vocab.numericalize(text)
        
        if len(numericalized_text) < self.max_length:
            numericalized_text.extend([0] * (self.max_length - len(numericalized_text)))
        else:
            numericalized_text = numericalized_text[:self.max_length]
            
        return torch.tensor(numericalized_text), torch.tensor(label, dtype=torch.float32)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True,
                           bidirectional=True)  
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  
    
    def forward(self, text):
        embedded = self.embedding(text)
        
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return torch.sigmoid(self.fc(hidden))

def calculate_class_weights(labels):
    total_samples = len(labels)
    n_samples_each_class = torch.bincount(labels.long())
    n_classes = len(n_samples_each_class)
    weights = total_samples / (n_classes * n_samples_each_class.float())
    return weights

def weighted_train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, class_weights, n_epochs):
    best_valid_loss = float('inf')
    spam_threshold = 0.5
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (text, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            text, labels = text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text).squeeze(1)
            
            # Apply class weights to the loss
            sample_weights = class_weights[labels.long()]
            losses = criterion(predictions, labels)
            loss = (losses * sample_weights).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (predictions > spam_threshold).float()  
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # validation
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        spam_correct = 0
        spam_total = 0
        
        with torch.no_grad():
            for text, labels in valid_loader:
                text, labels = text.to(device), labels.to(device)
                predictions = model(text).squeeze(1)
                
                sample_weights = class_weights[labels.long()]
                losses = criterion(predictions, labels)
                loss = (losses * sample_weights).mean()
                
                valid_loss += loss.item()
                predicted = (predictions > spam_threshold).float()  
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                
                spam_mask = (labels == 0)
                spam_total += spam_mask.sum().item()
                spam_correct += ((predicted == labels) & spam_mask).sum().item()
        
        valid_loss = valid_loss / len(valid_loader)
        valid_acc = valid_correct / valid_total
        spam_recall = spam_correct / spam_total if spam_total > 0 else 0
        
        scheduler.step(valid_loss)
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')
        print(f'\tSpam Recall: {spam_recall:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'spam_lstm_model.pt')

def main():
    X = email_ds['text']
    y = email_ds['label']
    
    class_counts = y.value_counts()
    print("\nClass Distribution:")
    print(f"Ham (1): {class_counts[1]}")
    print(f"Spam (0): {class_counts[0]}")
    print(f"Ratio (Ham/Spam): {class_counts[1]/class_counts[0]:.2f}\n")
    
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    valid_df = pd.DataFrame({'text': X_test, 'label': y_test})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    
    train_dataset = SpamDataset(train_df, vocab)
    valid_dataset = SpamDataset(valid_df, vocab)
    test_dataset = SpamDataset(test_df, vocab)
    
    train_labels = torch.tensor(y_train.values)
    class_weights = calculate_class_weights(train_labels).to(device)
    print("Class weights:", class_weights.cpu().numpy())
    
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 150
    HIDDEN_DIM = 256
    N_LAYERS = 1
    DROPOUT = 0.2
    
    model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    
    criterion = nn.BCELoss(reduction='none')  
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    weighted_train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, class_weights, n_epochs=10)
    
    model.load_state_dict(torch.load('spam_lstm_model.pt'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    spam_correct = 0
    spam_total = 0
    
    spam_threshold = 0.5  
    
    with torch.no_grad():
        for text, labels in test_loader:
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            predicted = (predictions > spam_threshold).float()  
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            spam_mask = (labels == 0)
            spam_total += spam_mask.sum().item()
            spam_correct += ((predicted == labels) & spam_mask).sum().item()
    
    test_acc = test_correct / test_total
    spam_recall = spam_correct / spam_total if spam_total > 0 else 0
    
    print(f'\nTest Results:')
    print(f'Overall Accuracy: {test_acc:.4f}')
    print(f'Spam Recall (% of spam caught): {spam_recall:.4f}')

if __name__ == "__main__":
    main()