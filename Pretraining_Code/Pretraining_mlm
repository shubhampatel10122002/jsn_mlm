!pip install transformers
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
import torch
import json
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# Load the CSV file
df = pd.read_csv("jsn_smiles.csv")
smiles_strings = df["SMILES_COLUMN"].tolist()

# Load custom vocab from the uploaded JSON file
with open("vocab.json", "r") as f:
    custom_vocab = json.load(f)

# Initialize a tokenizer with custom vocab
custom_tokenizer = RobertaTokenizerFast(
    vocab_file="vocab.json",
    merges_file="merges.txt",
    model_max_length=512,
)

# Tokenize SMILES strings
encoded_inputs = custom_tokenizer(
    smiles_strings,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",  # Return PyTorch tensors
)

# Define the excluded token IDs
excluded_token_ids = [591, 592, 593, 594]

# Create a mask to exclude specific token IDs from being masked
excluded_mask = torch.ones_like(encoded_inputs["input_ids"], dtype=torch.bool)
for token_id in excluded_token_ids:
    excluded_mask &= (encoded_inputs["input_ids"] != token_id)

# Mask a percentage of the tokens (15%) excluding the specified token IDs
rand = torch.rand(encoded_inputs["input_ids"].shape)
mask_ids = (rand < 0.15) & encoded_inputs["attention_mask"].bool() & excluded_mask

input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

# Create a copy of the input_ids to use for MLM prediction
labels = input_ids.clone()

# Apply the masking
input_ids[mask_ids] = custom_tokenizer.mask_token_id

encodings = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels,
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        return {
            key: tensor[i] for key, tensor in self.encodings.items()
        }


# Create DataLoader
dataset = Dataset(encodings)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Configure RoBERTa Model
config = RobertaConfig(
    vocab_size=596,
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

from transformers import AdamW

optim = AdamW(model.parameters(), lr=1e-4)

# Early Stopping Configuration
best_val_loss = float("inf")  # Initialize with a high value
early_stopping_patience = 5  # Number of epochs to wait before early stopping
epochs_without_improvement = 0
num_epochs = 100  # Define the maximum number of training epochs

# Validation Data (Split your data into training and validation sets)
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)  # Adjust the test_size as needed

# Tokenize and create DataLoader for validation data
val_smiles_strings = val_data["SMILES_COLUMN"].tolist()
val_encoded_inputs = custom_tokenizer(
    val_smiles_strings,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
val_input_ids = val_encoded_inputs["input_ids"]
val_attention_mask = val_encoded_inputs["attention_mask"]
val_labels = val_input_ids.clone()
val_encoded_inputs = {
    'input_ids': val_input_ids,
    'attention_mask': val_attention_mask,
    'labels': val_labels,
}
val_dataset = Dataset(val_encoded_inputs)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(dataloader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

        loop.set_description(f'Epoch: {epoch}')
        loop.set_postfix(loss=loss.item())

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_batch in val_dataloader:
            val_inputs = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['labels'].to(device)

            val_outputs = model(val_inputs, attention_mask=val_attention_mask, labels=val_labels)
            val_loss += val_outputs.loss.item()

        val_loss /= len(val_dataloader)

    print(f"Epoch {epoch}: Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model if desired
        model.save_pretrained("best_model")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered. Training halted.")
        break

# Save the final trained model
model.save_pretrained("harikrishna_jsn")
# Initialize variables for accuracy calculation
total_tokens = 0
correct_predictions = 0

# Loop through the batches in the DataLoader
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Generate predictions
    with torch.no_grad():
        predictions = model(input_ids, attention_mask=attention_mask)

    # Compare predictions with labels
    masked_tokens = input_ids == custom_tokenizer.mask_token_id
    correct_predictions += (predictions.logits.argmax(dim=-1) == labels).masked_select(masked_tokens).sum().item()
    total_tokens += masked_tokens.sum().item()

# Calculate accuracy
accuracy = (correct_predictions / total_tokens) * 100


