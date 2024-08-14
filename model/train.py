import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import AdamW
import os
import requests
from zipfile import ZipFile

# Set paths to custom word lists and data
word_lists_path = 'model/word-lists/'
stopwords_path = os.path.join(word_lists_path, 'stopwords.csv')
negative_words_path = os.path.join(word_lists_path, 'negative-words.csv')
positive_words_path = os.path.join(word_lists_path, 'positive-words.csv')

# Download and extract Glove embeddings file
glove_dir = 'model/word-lists/'
glove_filename = 'glove.6B.100d.txt'
glove_zip_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
glove_zip_path = os.path.join(glove_dir, 'glove.6B.zip')

# Ensure that Glove embeddings downloaded and available
if not os.path.exists(os.path.join(glove_dir, glove_filename)):
    # Download GloVe zip file
    print("Downloading GloVe embeddings...")
    response = requests.get(glove_zip_url, stream=True)
    # Show progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    downloaded_size = 0
    with open(glove_zip_path, 'wb') as f:
        for data in response.iter_content(block_size):
            downloaded_size += len(data)
            f.write(data)
            progress = downloaded_size / total_size * 100
            print(f"Download progress: {progress:.2f}%", end='\r')
    print("Download complete.")
    with ZipFile(glove_zip_path, 'r') as zip_ref:
        zip_ref.extract(glove_filename, path=glove_dir)
    # Clean up the zip file
    os.remove(glove_zip_path)
    print("GloVe embeddings downloaded and extracted.")
glove_path = os.path.join(glove_dir, glove_filename)

# Load stopwords from CSV
custom_stopwords = pd.read_csv(stopwords_path, encoding='latin1')["Word"].tolist()
# Load positive and negative word lists
positive_words = pd.read_csv(positive_words_path, header=None, encoding='latin1')[0].apply(lambda x: x.split('\t')[0]).tolist()
negative_words = pd.read_csv(negative_words_path, header=None, encoding='latin1')[0].apply(lambda x: x.split('\t')[0]).tolist()
# Load and preprocess the dataset
data_path = 'model/export-data/cleaned_and_padded_reviews.csv'
data = pd.read_csv(data_path)

# Function to remove stopwords from text
def preprocess_text(text, stopwords):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# Apply preprocessing to the review texts
data['padded_review_text'] = data['padded_review_text'].apply(lambda x: preprocess_text(x, custom_stopwords))

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Tokenize the texts and convert them to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['padded_review_text'].values)
sequences = tokenizer.texts_to_sequences(data['padded_review_text'].values)
word_index = tokenizer.word_index

# Pad sequences to ensure uniform input size
max_length = data['padded_review_text'].apply(lambda x: len(x.split())).max()
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
y = np.array(data['type'].values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load GloVe embeddings and create an embedding matrix
embedding_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the model architecture
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, input_length=max_length,
              weights=[embedding_matrix], trainable=False),  # Use pre-trained embeddings, frozen during training
    LSTM(128, return_sequences=True),  
    GlobalAveragePooling1D(),  
    Dense(128, activation='relu'),  
    Dropout(0.5),  
    Dense(1, activation='sigmoid')  
])

# Compile the model with AdamW optimizer
optimizer = AdamW(learning_rate=0.0005, weight_decay=1e-4)  # AdamW optimizer with weight decay
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks for early stopping and model checkpointing
model_checkpoint_path = 'model/trained-model/sentiment_model.keras'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),  # Stop training early if no improvement
    ModelCheckpoint(filepath=model_checkpoint_path, save_best_only=True, monitor='val_loss')  # Save the best model
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model and tokenizer
model.save('model/trained-model/sentiment_model_final.keras')
with open('model/trained-model/tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())

print("Model and tokenizer saved successfully.")
