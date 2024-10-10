# Named Entity Recognition (NER) for Mountain Names

This project fine-tunes a BERT-based model to recognize mountain names using a custom CoNLL format dataset. The model identifies mountain names from sentences, which can be useful in geographic or tourism-related applications. 

## Solution Explanation

The task is approached using a **fine-tuned BERT model** for Named Entity Recognition (NER). BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model known for its effectiveness in tasks like token classification, which includes NER.

### How the Model Works

1. **Dataset**: A CoNLL-formatted dataset containing mountain names tagged as entities is used. Each word in a sentence is tagged with a label indicating whether it is a beginning or inside of a mountain entity (`B-LOC` or `I-LOC`) or not an entity (`O`).
   
2. **BERT Model**: The project uses a pre-trained `bert-base-cased` model, which is fine-tuned for the task. BERT is well-suited for token-level tasks like NER due to its ability to understand context bidirectionally (i.e., both left-to-right and right-to-left).

3. **Tokenization**: The BERT tokenizer splits the sentences into tokens and aligns the labels with the tokenized data. Subword tokens that appear due to tokenization are ignored during label alignment, ensuring that predictions are only made on meaningful tokens.

4. **Training**: Using Hugging Face's `Trainer`, the BERT model is fine-tuned with the labeled dataset, and the weights are adjusted for the specific NER task. The model is trained over 3 epochs with a learning rate of `2e-5` and weight decay of `0.01` for regularization.

5. **Inference**: After training, the model is used to predict entities in new sentences, identifying mountain names based on the trained entity labels.

### Project Structure

- **train_model.py**: Script to fine-tune the BERT model on the dataset.
- **inference_model.py**: Script to perform inference and predict mountain names in new sentences.
- **labeled_mountains_dataset.conll**: Custom dataset in CoNLL format used for training.
- **model_weights.zip**: Pre-trained model weights used for inference (to be downloaded separately).

