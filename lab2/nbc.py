import os
import json
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

# --- Main config ---
BOOK_DIR = "harry_potter"
PAGE_SIZE = 400
EMBEDDING_DIM = 50
LAB1_MODEL_PATH = "word2vec_model_A.pth"
LAB1_VOCAB_PATH = "word_to_idx_A.json"
# CNN Hyper params
NUM_FILTERS = 100
KERNEL_SIZE = 3
NUM_CLASSES = 7
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DROPOUT_PROB = 0.5
WEIGHT_DECAY = 1e-4

# --- GPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- System will be using device: {device} ---")

def prepare_dataset(book_dir, page_size):
    print ("Starting data prep...")
    all_pages = []
    all_labels = []

    for i in range(1, 8):
        file_path = os.path.join(book_dir, f"HP{i}.txt")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except FileNotFoundError:
            print(f"Cant find file in {file_path}")
            continue
        print(f"Processing {file_path}...")

        clean_text = full_text.lower()
        punctuation_list = string.punctuation + '“”’'
        translator = str.maketrans('', '', punctuation_list)
        clean_text = clean_text.translate(translator)
        tokens = clean_text.split()

        num_pages_in_book = len(tokens) // page_size
        for j in range(num_pages_in_book):
            start_index = j * page_size
            end_index = start_index + page_size
            page = tokens[start_index:end_index]
            
            all_pages.append(page)
            # The label is i-1 to create labels from 0 to 6.
            all_labels.append(i - 1) 

    print(f"\nData preparation complete.")
    return all_pages, all_labels

def embed_pages(pages, embedding_matrix, word_to_idx, page_size, embedding_dim):
    print("\nEmbedding pages with the Lab 1 encoder...")
    # We will create a 3D numpy array to hold all the embedded pages.
    # The shape will be (number_of_pages, words_per_page, embedding_dimension)
    embedded_pages = np.zeros((len(pages), page_size, embedding_dim))

    for i, page in enumerate(pages):
        for j, word in enumerate(page):
            # Look up the word's index in our Lab 1 vocabulary.
            word_index = word_to_idx.get(word)
            
            # If the word exists in our vocabulary, get its embedding.
            if word_index is not None:
                embedded_pages[i, j] = embedding_matrix[word_index]
            # If the word is not in our vocabulary, it will remain a vector of zeros.
            
    print("Embedding complete.")
    return embedded_pages

class CNNClassifier(nn.Module):
    # https://www.geeksforgeeks.org/nlp/text-classification-using-cnn/
    # https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f/
    def __init__(self, embedding_dim, num_filters, kernel_size, num_classes, dropout_prob):
        super(CNNClassifier, self).__init__()
        # The convolutional layer expects input of shape (batch_size, in_channels, sequence_length)
        # We have to reshape our embedded pages to match this.
        # Here, in_channels is the embedding dimension.
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        # Max-pooling will take the maximum value from the output of the conv layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x starts as (batch_size, page_size, embedding_dim)
        # We need to swap the last two dimensions for the conv layer
        x = x.permute(0, 2, 1) # -> (batch_size, embedding_dim, page_size)
        
        x = self.conv1d(x) # -> (batch_size, num_filters, page_size - kernel_size + 1)
        x = self.relu(x)
        
        x = self.pool(x) # -> (batch_size, num_filters, 1)
        x = x.squeeze(2) # -> (batch_size, num_filters)
        x = self.dropout(x)
        
        x = self.fc(x) # -> (batch_size, num_classes)
        return x

def predict_random_page(model, val_data, val_labels):
    """
    Selects a random page from the validation set and displays the model's prediction.
    """
    model.eval() # Ensure the model is in evaluation mode
    
    # Select a random index from the validation set
    random_idx = random.randint(0, len(val_data) - 1)
    sample_page = val_data[random_idx]
    true_label_idx = val_labels[random_idx].item()

    # The model expects a batch, so we add a batch dimension of 1
    sample_page_batch = sample_page.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_page_batch)
        _, predicted_idx = torch.max(output.data, 1)
        predicted_idx = predicted_idx.item()

    # Convert indices back to human-readable book numbers
    true_book = f"Book {true_label_idx + 1}"
    predicted_book = f"Book {predicted_idx + 1}"

    print("\n--- Single Page Prediction ---")
    print(f"Selected a random page from the validation set (Index: {random_idx}).")
    print(f"The page is actually from: {true_book}")
    print(f"The model predicted: {predicted_book}")
    if true_book == predicted_book:
        print("Result: Correct ✅")
    else:
        print("Result: Incorrect ❌")

if __name__ == "__main__":
    pages, labels = prepare_dataset(BOOK_DIR, PAGE_SIZE)

    if pages:
        print(f"Total number of pages created: {len(pages)}")
        print(f"Total number of labels: {len(labels)}")
        
        try:
            lab1_model_state = torch.load(LAB1_MODEL_PATH, map_location=torch.device('cpu'))
            embedding_matrix = lab1_model_state['embeddings.weight'].numpy()
            
            with open(LAB1_VOCAB_PATH, 'r') as f:
                word_to_idx = json.load(f)

            X = embed_pages(pages, embedding_matrix, word_to_idx, PAGE_SIZE, EMBEDDING_DIM)
            y = np.array(labels)

            print(f"\nFinal shape of the input data (X): {X.shape}")
            print(f"Final shape of the labels (y): {y.shape}")

            print("\nSplitting data into training and validation sets...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Convert numpy arrays to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            # Create DataLoaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            print("Data splitting complete.")

            # --- Initialize and Train the CNN Model ---
            print("\n--- Training Base CNN Classifier ---")
            model = CNNClassifier(EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZE, NUM_CLASSES, DROPOUT_PROB).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # --- Evaluate on Validation Set ---
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                accuracy = 100 * correct / total
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")
            predict_random_page(model, X_val_tensor, y_val_tensor)

        except FileNotFoundError:
            print(f"\nError: Could not find Lab 1 model ('{LAB1_MODEL_PATH}') or vocabulary ('{LAB1_VOCAB_PATH}').")
            print("Please ensure you have run Lab 1 and saved these files.")
        except Exception as e:
            print(f"\nAn error occurred during the embedding step: {e}")
