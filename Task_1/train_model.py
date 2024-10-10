# train_model.py
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch

# Function to parse the CoNLL dataset
def parse_conll(data_file):
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(data_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue
            word, tag = line.split()
            current_sentence.append(word)
            current_labels.append(tag)
    
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    
    return sentences, labels

# Parse the dataset
sentences, labels = parse_conll('labeled_mountains_dataset.conll')

# Prepare the dataset for BERT tokenization
data = {"tokens": sentences, "ner_tags": labels}
dataset = Dataset.from_dict(data)

# Load pre-trained tokenizer and define label list
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
label_list = ["O", "B-LOC", "I-LOC"]
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list))

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        max_length=128,
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_list.index(label[word_idx]))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./model_weights")
