import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn
import os
from transformers import RobertaConfig
import numpy as np

# Loading data
# data_path = '/content/drive/MyDrive/JSN_MTR/Harikrishna_uniform.csv'
data_path = '/content/drive/MyDrive/Harikrishna_Data/Harikrishna_March1_data_strict.csv'

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

tokenizer = RobertaTokenizer.from_pretrained('/content/drive/MyDrive/JSN_mlm/')
roberta_model = RobertaForSequenceClassification.from_pretrained('/content/drive/MyDrive/JSN_mlm/', num_labels=1)

# Define additional layers
class CustomSurfaceTensionModel(nn.Module):
    def __init__(self, roberta_model):
        super(CustomSurfaceTensionModel, self).__init__()
        self.roberta = roberta_model.roberta
        self.classifier = nn.Sequential(
            nn.Linear(384, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(512, 512),#
            # nn.ReLU(),#
            # nn.Dropout(0.2),#
            nn.Linear(512, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

model = CustomSurfaceTensionModel(roberta_model)

# Set up data loaders
train_dataset = SurfaceTensionDataset(train_df, tokenizer)
val_dataset = SurfaceTensionDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epoch = 150

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=4.72e-5)
total_steps = len(train_loader) * epoch
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Set up loss function for regression
criterion = nn.SmoothL1Loss()

# Training loop
for epoch in range(epoch):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.flatten(), labels)
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epoch}, Training Loss: {average_loss}')

# Save the trained model
torch.save(model.state_dict(), 'surface_tension_model_1.pth')

# Save the tokenizer
tokenizer_save_path = '/content/tokenizer'
tokenizer.save_pretrained(tokenizer_save_path)
print(f'Tokenizer saved at: {tokenizer_save_path}')

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
        predictions.extend(outputs.flatten().cpu().numpy())
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

# Load the trained model for prediction
loaded_model = CustomSurfaceTensionModel(roberta_model)
loaded_model.load_state_dict(torch.load('surface_tension_model.pth'))
loaded_model.to(device)
loaded_model.eval()

# Tokenize the input SMILES string "O"
input_smiles = "O"
encoding = tokenizer(
    input_smiles,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors='pt'
)

# Make predictions
with torch.no_grad():
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Model prediction
    output = loaded_model(input_ids, attention_mask=attention_mask)
    predicted_value = output.flatten().item()

# Inverse transform the predicted value to get back to the original scale
predicted_value = scaler.inverse_transform(np.array([[predicted_value]]))[0, 0]

print(f'Predicted Surface Tension for "O": {predicted_value}')
