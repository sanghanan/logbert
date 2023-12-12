import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import label_binarize

# Constants
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 64
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
EPOCHS = 100
PATIENCE = 3
NUM_LABELS = 2
DATA_FILE = 'filtered.csv'

# Function to encode the dataset
def encode_examples(df, tokenizer, max_length=MAX_LENGTH):
    input_ids = []
    attention_masks = []
    labels = []

    for _, row in df.iterrows():
        bert_input = tokenizer.encode_plus(
            row['static_text'],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])
        labels.append(row['log_level_encoded'])

    return (tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks)), tf.convert_to_tensor(labels)

# Function to create TensorFlow dataset
def create_dataset(data, labels, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(len(labels)).batch(batch_size)
    return dataset

# Function to load and process the dataset
def load_and_process_data(file_name):
    df = pd.read_csv(file_name)
    train_df, test_df = train_test_split(df, test_size=TRAIN_TEST_SPLIT)
    train_data, train_labels = encode_examples(train_df, tokenizer)
    test_data, test_labels = encode_examples(test_df, tokenizer)
    return create_dataset(train_data, train_labels), create_dataset(test_data, test_labels), test_df


# Function to compile and train the model
def compile_and_train_model(model, train_dataset, test_dataset, epochs=EPOCHS, patience=PATIENCE):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, mode='min', restore_best_weights=True)

    model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=[early_stopping])

def encode_examples_for_prediction(df, tokenizer, max_length=MAX_LENGTH):
    input_ids = []
    attention_masks = []

    for _, row in df.iterrows():
        bert_input = tokenizer.encode_plus(
            row['static_text'],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks)

def predict_and_evaluate(model, data_df, tokenizer):
    # Encode the examples
    encoded_data = encode_examples_for_prediction(data_df, tokenizer)
    # Predict
    predictions = model.predict(encoded_data)
    predicted_labels = np.argmax(predictions.logits, axis=1)



    # True Labels
    true_labels = data_df['log_level_encoded'].values

    # Identify Misclassified
    # Identify Misclassified
    misclassified_indices = predicted_labels != true_labels
    misclassified = data_df[misclassified_indices].copy()
    misclassified['predicted_labels'] = predicted_labels[misclassified_indices]

    # Calculate Accuracy and AUC
    accuracy = accuracy_score(true_labels, predicted_labels)
    # Assuming binary classification for AUC, adjust if you have multi-class
    auc = roc_auc_score(true_labels, predictions.logits[:,1])

    # Calculate AUC for Multiclass
    # y_true_binarized = label_binarize(true_labels, classes=np.unique(true_labels))
    # auc = roc_auc_score(y_true_binarized, predictions.logits, multi_class='ovr')

    return predictions, misclassified, accuracy, auc


def main():
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Process initial dataset
    train_dataset, test_dataset, test_df = load_and_process_data(DATA_FILE)

    # Model Configuration and Training
    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    compile_and_train_model(model, train_dataset, test_dataset)

    # Save the model
    model.save_pretrained('model')

    # Predict and Evaluate
    predictions, misclassified, accuracy, auc = predict_and_evaluate(model, test_df, tokenizer)

    # Print results
    print("AUC:", round(auc, 2))
    print("Accuracy:", round(accuracy, 2))
    misclassified.to_csv("misclassified.csv")

if __name__ == "__main__":
    main()