from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, PreTrainedTokenizerFast
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import notebook_launcher

# transformers/models/opt/modeling_opt.py:OPTForCausalLM:forward
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m",
                                             device_map="cuda")

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("facebook/opt-350m")

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
dataset = dataset.train_test_split(test_size=0.999)["train"]

collator = DataCollatorForCompletionOnlyLM(instruction_template="### Human:",
                                           response_template="### Assistant:",
                                           ignore_index=tokenizer.pad_token_id,
                                           tokenizer=tokenizer)

dataset_text_field = "text"
learning_rate = 1.41e-5
batch_size = 1
seq_length = 128
gradient_accumulation_steps = 16
num_train_epochs = 3

training_args = TrainingArguments(
    output_dir="./oasst_sft_opt",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    logging_steps=1,
    num_train_epochs=num_train_epochs,
    report_to="none",
    save_total_limit=5,
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=seq_length,
    train_dataset=dataset,
    dataset_text_field=dataset_text_field,
    data_collator=collator,
)

trainer.train()
