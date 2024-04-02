import argparse
from transformers import MT5ForSequenceClassification, MT5ForConditionalGeneration, AutoTokenizer

def upload_model_to_hub(local_model_path, model_name, model_type):
    if model_type == 'seq_class':
        # Load the local model and tokenizer
        model = MT5ForSequenceClassification.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # Upload the model to the Hugging Face Model Hub
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)
    elif model_type == 'cond_gen':
        # Load the local model and tokenizer
        model = MT5ForConditionalGeneration.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # Upload the model to the Hugging Face Model Hub
        model.push_to_hub(model_name)
        tokenizer.push_to_hub(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a HuggingFace Transformer model to the Model Hub")
    parser.add_argument("--model_path", required=True, help="Path to the local model directory")
    parser.add_argument("--model_name", required=True, help="Name for the model on the Model Hub")
    parser.add_argument("--model_type", required=True, help="Either seq_class or cond_gen")

    args = parser.parse_args()

    upload_model_to_hub(args.model_path, args.model_name, args.model_type)

# Usage: python push-to-hub.py --model_path ../finetuned-models/output --model_name adenhaus/mt5-small-tata --model_type cond_gen
