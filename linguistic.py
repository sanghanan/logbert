import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, f1_score
import pandas as pd
import numpy as np

# Constants
FILE_PATH = 'lingustic_quality_inter.csv'
BERT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 64
TEST_SIZE = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 100
PATIENCE = 3

def load_data(filepath):
    """
    Load data from a CSV file.

    Args:
    filepath (str): Path to the CSV file.

    Returns:
    DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def encode_examples(df, tokenizer, max_length=64):
    """
    Encode examples for BERT input.

    Args:
    df (DataFrame): Data containing text and labels.
    tokenizer: BERT tokenizer.
    max_length (int): Maximum length for tokenization.

    Returns:
    Tuple: Encoded inputs and labels.
    """
    input_ids, attention_masks, labels = [], [], []
    for _, row in df.iterrows():
        bert_input = tokenizer.encode_plus(
            row['log_messages'], 
            add_special_tokens=True, 
            max_length=max_length, 
            pad_to_max_length=True, 
            return_attention_mask=True
        )
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])
        labels.append(row['label'])
    return (input_ids, attention_masks), labels

def create_tf_dataset(data, labels, batch_size=32, shuffle_size=None):
    """
    Create TensorFlow dataset.

    Args:
    data: Input data.
    labels: Corresponding labels.
    batch_size (int): Batch size for the dataset.
    shuffle_size (int): Size for shuffling the dataset.

    Returns:
    TensorFlow Dataset: Prepared dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle_size:
        dataset = dataset.shuffle(shuffle_size)
    return dataset.batch(batch_size)

def build_and_compile_model():
    """
    Build and compile the BERT model.

    Returns:
    Model: Compiled BERT model.
    """
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=2)
    model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def get_label(num):
    """
    Convert numerical label to textual label.

    Args:
    num (int): Numerical label.

    Returns:
    str: Textual representation of the label.
    """
    return "insufficient" if num == 0 else "sufficient"

def evaluate_model(model, test_dataset):
    """
    Evaluate the model and print precision and F1 score.

    Args:
    model: Trained model.
    test_dataset: Dataset for evaluation.
    """
    test_predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(test_predictions.logits, axis=1)
    actual_labels = np.concatenate([y for _, y in test_dataset], axis=0)

    precision = precision_score(actual_labels, predicted_labels, average='weighted')
    f1 = f1_score(actual_labels, predicted_labels, average='weighted')

    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")

def main():
    # Load and preprocess data
    df = load_data(FILE_PATH)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE)
    train_data, train_labels = encode_examples(train_df, tokenizer)
    test_data, test_labels = encode_examples(test_df, tokenizer)

    # Create datasets
    train_dataset = create_tf_dataset(train_data, train_labels, shuffle_size=len(train_df))
    test_dataset = create_tf_dataset(test_data, test_labels)

    # Build and train the model
    model = build_and_compile_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', restore_best_weights=True)
    model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=[early_stopping])

    # Evaluate the model
    evaluate_model(model, test_dataset)

if __name__ == "__main__":
    main()
