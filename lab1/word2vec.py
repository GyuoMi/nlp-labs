import numpy as np
import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib.pyplot as plt

# --- Main Configuration ---
BOOK_FILE_PATH = "harry_potter/HP4.txt"
CONTEXT_WINDOW_SIZE = 2
EMBEDDING_DIM = 50 
EPOCHS = 50
LEARNING_RATE = 0.025
BATCH_SIZE = 1024
INIT_STD = 0.01
VOCAB_SIZE = 10000

# --- GPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- System will be using device: {device} ---")


# --- Data Preparation Function ---

def prepare_data(full_text, part, context_window_size, vocab_size):
    """Prepares a chunk of text for Word2Vec training."""
    print(f"--- Preparing data for Part {part} ---")

    clean_text = full_text.lower()
    punctuation_list = string.punctuation + '“”’'
    translator = str.maketrans('', '', punctuation_list)
    clean_text = clean_text.translate(translator)
    tokens = clean_text.split()
    print(f"Tokens preview (before filtering): {tokens[:20]}")

    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 
        'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
        'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
    }
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords]
    print(f"Tokens preview (after filtering): {tokens[:20]}")
    
    word_counts = Counter(tokens)
    vocabulary = [word for word, count in word_counts.most_common(vocab_size)]
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    print(f"Vocabulary preview: {vocabulary[:20]}")
    print(f"Vocabulary size: {len(vocabulary)}")
    
    training_pairs = []
    tokens_in_vocab = [word for word in tokens if word in word_to_idx]
    for i, focal_word in enumerate(tokens_in_vocab):
        focal_idx = word_to_idx[focal_word]
        for j in range(max(0, i - context_window_size), min(len(tokens_in_vocab), i + context_window_size + 1)):
            if i == j:
                continue
            context_word = tokens_in_vocab[j]
            context_idx = word_to_idx[context_word]
            training_pairs.append((focal_idx, context_idx))
            
    print(f"Number of training pairs: {len(training_pairs)}\n")
    print("Training pairs (focal_word -> context_word):")
    print("\nFirst 5 pairs only:")
    for focal_idx, context_idx in training_pairs[:10]:
        print(f"{tokens[focal_idx]} ({focal_idx}) -> {tokens[context_idx]} ({context_idx})")
    return vocabulary, word_to_idx, training_pairs

# --- Model Training Function ---

def train_model(vocabulary, training_pairs, embedding_dim, epochs, learning_rate, batch_size, init_std, part):
    """Initialises and trains a SkipGramModel, returning the learned embeddings and loss history."""
    vocab_size = len(vocabulary)
    loss_history = []

    X_train = torch.tensor([center for center, _ in training_pairs], dtype=torch.long)
    y_train = torch.tensor([context for _, context in training_pairs], dtype=torch.long)
    dataset = TensorDataset(X_train, y_train)
#     gpu optimisations to parellise code better and improve cpu to gpu instruction transfer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = SkipGramModel(vocab_size, embedding_dim, init_std).to(device)
    criterion = nn.NLLLoss()
#     improvement over sgd
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True) #non blocking helps keep gpu busy while cpu loading instructions
            batch_y = batch_y.to(device, non_blocking=True)

            optimiser.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
            
    model_save_path = f"word2vec_model_{part}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for Part {part} saved to {model_save_path}")
#   learned embeddings from the embedding layer
    return model.embeddings.weight.data, loss_history

# --- Plotting Function ---

def plot_and_save_loss_curves(loss_A, loss_B, filename="loss_curve_graph.png"):
    """Takes loss histories and saves a plot to a file."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_A, label="Model A Loss (First Half)")
    plt.plot(loss_B, label="Model B Loss (Second Half)")
    plt.title("Training Loss Curves per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    print(f"\nLoss curve graph saved to {filename}")

# --- Inference function ---
def word_to_one_hot(word, word_to_idx):
    vocab_size = len(word_to_idx)
    one_hot = torch.zeros(vocab_size)
    idx = word_to_idx.get(word)
    if idx is None:
        raise ValueError(f"Word '{word}' not found in vocabulary.")
    one_hot[idx] = 1
    return one_hot

def get_embedding_from_word(model, word, word_to_idx):
    """retrieves the embedding for a word directly from a trained model."""
    word_index = word_to_idx.get(word)
    if word_index is None:
        return None
    
    # The fix is to add .to(device) here, moving the input tensor to the GPU
    word_tensor = torch.tensor([word_index]).to(device)
    
    embedding = model.embeddings(word_tensor)
    return embedding.squeeze()

# --- Model Definition ---

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, init_std):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.output.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.output(x)
        x = self.log_softmax(x)
        return x

# --- Main Execution Block ---

if __name__ == "__main__":
    
    print("=== Loading and Splitting Data ===")
    with open(BOOK_FILE_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    split_point = len(full_text) // 2
    text_A = full_text[:split_point]
    text_B = full_text[split_point:]
    print(f"Split book into two halves. Part A has {len(text_A)} characters, Part B has {len(text_B)} characters.\n")

    print("=== Training Model A (First Half) ===")
    vocab_A, word_to_idx_A, pairs_A = prepare_data(text_A, 'A', CONTEXT_WINDOW_SIZE, VOCAB_SIZE)
    embeddings_A, loss_history_A = train_model(vocab_A, pairs_A, EMBEDDING_DIM, EPOCHS, LEARNING_RATE, BATCH_SIZE, INIT_STD, 'A')
    
    print("\n=== Training Model B (Second Half) ===")
    vocab_B, word_to_idx_B, pairs_B = prepare_data(text_B, 'B', CONTEXT_WINDOW_SIZE, VOCAB_SIZE)
    embeddings_B, loss_history_B = train_model(vocab_B, pairs_B, EMBEDDING_DIM, EPOCHS, LEARNING_RATE, BATCH_SIZE, INIT_STD, 'B')
    
    plot_and_save_loss_curves(loss_history_A, loss_history_B)
    
    print("\n=== Learning and Evaluating the Mapping ===")
    
    embeddings_A_np = embeddings_A.cpu().numpy()
    embeddings_B_np = embeddings_B.cpu().numpy()
    
    shared_vocab = sorted(list(set(word_to_idx_A.keys()).intersection(set(word_to_idx_B.keys()))))
    print(f"Found {len(shared_vocab)} shared words between the two halves.")

    X_A = np.zeros((len(shared_vocab), EMBEDDING_DIM))
    X_B = np.zeros((len(shared_vocab), EMBEDDING_DIM))

    for i, word in enumerate(shared_vocab):
        X_A[i] = embeddings_A_np[word_to_idx_A[word]]
        X_B[i] = embeddings_B_np[word_to_idx_B[word]]

    T = np.linalg.pinv(X_A) @ X_B
    print(f"Learned transformation matrix T with shape: {T.shape}")
    
    mapped_X_A = X_A @ T
    dot_products = np.sum(mapped_X_A * X_B, axis=1)
    norms_A = np.linalg.norm(mapped_X_A, axis=1)
    norms_B = np.linalg.norm(X_B, axis=1)
    
    valid_indices = (norms_A > 0) & (norms_B > 0)
    similarities = dot_products[valid_indices] / (norms_A[valid_indices] * norms_B[valid_indices])
    
    average_similarity = np.mean(similarities)
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Average Cosine Similarity: {average_similarity:.4f}")
    
    print("\n--- Example Word-level Similarities ---")
    for i in range(min(15, len(shared_vocab))):
        word = shared_vocab[i]
        similarity = similarities[i]
        print(f"Word: '{word}', Mapped Similarity: {similarity:.4f}")

    print("\n=== Inference Demonstration ===")
    
    # We need a model object to perform inference. Let's reload Model A.
    # First, we need to initialize a new model with the correct dimensions.
    model_A_for_inference = SkipGramModel(len(vocab_A), EMBEDDING_DIM, INIT_STD).to(device)
    
    # Now, we load the saved weights
    model_A_for_inference.load_state_dict(torch.load("word2vec_model_A.pth"))
    model_A_for_inference.eval() # Set the model to evaluation mode

    example_words = ['harry', 'ron', 'hermione', 'dumbledore', 'magic']

    for word in example_words:
        print(f"\n--- Inferring for word: '{word}' ---")
        try:
            # Demonstrate getting the one-hot vector
            one_hot = word_to_one_hot(word, word_to_idx_A)
            if one_hot is not None:
                # We won't print the full vector as it's very long, just confirm its creation
                print(f"Successfully created a 1-hot vector of shape: {one_hot.shape}")
            else:
                print(f"Word '{word}' not found in Model A's vocabulary.")
                continue

            # Demonstrate getting the corresponding embedding
            embedding = get_embedding_from_word(model_A_for_inference, word, word_to_idx_A)
            if embedding is not None:
                # Print the first 5 dimensions of the embedding vector as a preview
                print(f"Embedding vector (first 5 dims): {embedding.detach().cpu().numpy()[:5]}")

        except Exception as e:
            print(f"An error occurred for word '{word}': {e}")
