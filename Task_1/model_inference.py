from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Path to the model weights in Google Drive
model_weights_path = 'model_weights/'

# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(model_weights_path)
model = BertForTokenClassification.from_pretrained(model_weights_path)

# Define label list
label_list = ["O", "B-LOC", "I-LOC"]

# Function to get NER predictions on a sentence
def predict_ner(sentence):
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Get predictions
    outputs = model(**inputs).logits

    # Get predicted labels
    predictions = outputs.argmax(dim=2)

    # Convert predictions to labels
    predicted_labels = [label_list[pred] for pred in predictions[0].tolist()]

    # Tokenized words
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Print the words along with their predicted labels
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")

# Test the model on a new sentence
if __name__ == "__main__":
    sentence = "Mount Everest is the highest peak in the world"
    predict_ner(sentence)
