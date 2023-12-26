
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your data
data_path = '/content/drive/MyDrive/JSN_MTR/jsn - smile-surface_tension - Sheet1.csv'
df = pd.read_csv(data_path)

# Scale and normalize the target variable
scaler = StandardScaler()
df['surface_tension'] = scaler.fit_transform(df[['surface_tension']])

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

class SurfaceTensionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = str(self.data.iloc[idx]['smiles'])
        label = float(self.data.iloc[idx]['surface_tension'])

        encoding = self.tokenizer(
            smile,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

tokenizer = RobertaTokenizer.from_pretrained('/content/drive/MyDrive/JSN_MTR/')
model = RobertaForSequenceClassification.from_pretrained('/content/drive/MyDrive/JSN_MTR/', num_labels=1)

# Freezing the RoBERTa base weights
for param in model.roberta.parameters():
    param.requires_grad = False

train_dataset = SurfaceTensionDataset(train_df, tokenizer)
val_dataset = SurfaceTensionDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 75
lr = 4.75e-5

optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}')

# Save the fine-tuned model
# save_path = '/content/fine_tuned_model/'
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

import numpy as np

# Validation loop
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions.extend(outputs.logits.flatten().cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Convert predictions and true_labels to NumPy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Inverse transform the predicted
# and true labels to get back to the original scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
true_labels = scaler.inverse_transform(true_labels.reshape(-1, 1)).flatten()

mse = mean_squared_error(true_labels, predictions)
print(f'Mean Squared Error on Validation Set: {mse}')

# Example of making predictions
smiles_to_predict = "O"
tokenized_input = tokenizer(smiles_to_predict, return_tensors='pt')
input_ids = tokenized_input['input_ids'].to(device)
attention_mask = tokenized_input['attention_mask'].to(device)

model.eval()
with torch.no_grad():
    prediction = model(input_ids, attention_mask=attention_mask).logits.item()

# Inverse transform the predicted label
# to get back to the original scale
prediction = scaler.inverse_transform([[prediction]])[0][0]

print(f'Predicted Surface Tension for SMILES "{smiles_to_predict}": {prediction}')
