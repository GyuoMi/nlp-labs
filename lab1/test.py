import string
import numpy as np
import os
import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt

DATA_FILE_NAME = "harry_potter/HP1.txt"
MODEL_NAME = "autoencoder_state.pth"
DELIMITER = " "

np.printoptions(threshold=np.inf)


class WordEncoder(torch.nn.Module):
    def __init__(self, K, num_words):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_words, K),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(K, num_words),
            torch.nn.Softmax(dim=1),
        )

        self.double()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Read one word on either side
WINDOW_SIZE = 1
K = 20
WORD_LIMIT = 200


def read_dataset():
    punctuation_translation = str.maketrans("", "", string.punctuation)
    words = []

    with open(DATA_FILE_NAME) as file:
        for line in file:
            if len(words) >= WORD_LIMIT:
                break

            for word in line.split():
                word = word.translate(punctuation_translation)
                words.append(word)

    unique_words = np.unique(words)

    vocab = {word: index for index, word in enumerate(unique_words)}
    num_classes = len(vocab)

    for key, value in vocab.items():
        vocab[key] = functional.one_hot(torch.tensor(value), num_classes)
        vocab[key] = vocab[key].numpy()

    return words, vocab, unique_words


def get_training_data(words, vocab):
    num_words = len(words)
    data_input, data_output = [], []

    for index in range(num_words):
        left_index = max(0, index - WINDOW_SIZE)
        right_index = min(num_words, index + WINDOW_SIZE + 1)

        for window_index in range(left_index, right_index):
            if window_index == index:
                continue

            input_word = words[index]
            output_word = words[window_index]

            input_tensor = vocab[input_word]
            output_tensor = vocab[output_word]

            data_input.append(input_tensor)
            data_output.append(output_tensor)

    return np.array([data_input, data_output], dtype=float)


def train_network(training_data, num_words):
    LEARNING_RATE = 1e-1
    WEIGHT_DECAY = 1e-8
    BATCH_SIZE = 15

    loader = torch.utils.data.DataLoader(
        dataset=training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = WordEncoder(K, num_words)

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    if os.path.exists(MODEL_NAME):
        model.load_state_dict(torch.load(MODEL_NAME))

    epochs = 500

    for epoch in range(epochs):
        for input_data, output_data in loader:
            model_output = model(input_data)
            loss = loss_function(model_output, output_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), MODEL_NAME)


def test_model(word, vocab_tensor, unique_words):
    num_words = len(vocab_tensor)
    model = WordEncoder(K, num_words)

    if os.path.exists(MODEL_NAME):
        model.load_state_dict(torch.load(MODEL_NAME))

    model.eval()

    input_data = np.array([vocab_tensor[word]])
    input_data = torch.tensor(input_data, dtype=torch.float64)

    output = model(input_data)
    output = output.detach().numpy()[0]
    word_index = np.argmax(output)
    print("\{" + word + ",", unique_words[word_index] + "\}", end=", ")


def main():
    words, vocab_tensor, unique_words = read_dataset()
    num_words = len(vocab_tensor)

    training_data = get_training_data(words, vocab_tensor)
    train_network(training_data, num_words)

    for word in unique_words:
        test_model(word, vocab_tensor, unique_words)


if __name__ == "__main__":
    main()
