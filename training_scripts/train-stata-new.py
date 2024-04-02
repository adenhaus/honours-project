import pandas as pd
import torch
from datasets import load_dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Load CSV datasets
datasets = load_dataset('csv', data_files={
    'train': '/disk/scratch/s2029717/data/train_stata.csv',
    'dev': '/disk/scratch/s2029717/data/dev_stata.csv', 
    'test': '/disk/scratch/s2029717/data/test_stata.csv'
})

# tokenizer = MT5Tokenizer.from_pretrained("google/mt5-xxl")
tokenizer = MT5Tokenizer.from_pretrained("/disk/scratch/s2029717/mt5-xl")
extra_token = "<extra_id_1>" 
tokenizer.add_tokens(extra_token)

# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-xxl")
model = MT5ForConditionalGeneration.from_pretrained("/disk/scratch/s2029717/mt5-xl")
model.resize_token_embeddings(len(tokenizer))  # Extra token

def preprocess_function(examples):
    sources = examples['full_input']
    targets = examples['attributable']
    model_inputs = tokenizer(sources, max_length=2048, truncation=True)

    # with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=512, truncation=True)
    
    labels = [[int(label)] for label in examples["attributable"]]

    model_inputs["labels"] = labels
    return model_inputs


tokenized_datasets = datasets.map(preprocess_function, batched=True)

class RegressionLogitsProcessor(torch.nn.Module):
    def __init__(self, extra_token_id):
        super().__init__()
        self.extra_token_id = extra_token_id

    def __call__(self, input_ids, scores):
        extra_token_logit = scores[:, :, self.extra_token_id] 
        return extra_token_logit


class RegressionTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        logits_processor = RegressionLogitsProcessor(tokenizer.get_vocab()[extra_token])
        regression_logits = logits_processor(inputs.input_ids, logits)

        loss_fct = torch.nn.MSELoss()

        loss = loss_fct(regression_logits.squeeze(-1), labels.squeeze(-1))

        rmse = torch.sqrt(loss)

        return (rmse, outputs) if return_outputs else rmse


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, max_length=2048)

training_args = Seq2SeqTrainingArguments(
    output_dir="/disk/scratch/s2029717/output-stata",
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    do_eval=True,
    learning_rate=0.0001,
    logging_steps=100,
    save_steps=100,
    eval_steps=100,
    lr_scheduler_type='constant',
    overwrite_output_dir=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=4,
    bf16=True,
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train() 
