import gc
import os

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_xpu_available, is_torch_npu_available
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Specify the checkpoint for SmolLM2
CHECKPOINT = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Define output directory based on the checkpoint name
OUTPUT_DIR = CHECKPOINT.split("/")[-1] + "-structure-output"
# Define template for prompts
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
INSTRUCTION_TEMPLATE = "<|im_start|>user\n"
PROMPT_TEMPLATE = """Query: {query}

schema:
{schema}"""


def clear_hardwares():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def main():

    # Load dataset and perform Train-Test Split
    ds = load_dataset("ChristianAzinn/json-training")
    split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_ds["train"]
    test_dataset = split_ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT,
                                                 device_map={"": PartialState().process_index}, )
    # Data collator for training
    collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE,
            instruction_template=INSTRUCTION_TEMPLATE,
            tokenizer=tokenizer,
            mlm=False
            )

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["query"])):
            messages = [
                {
                    "role":    "system",
                    "content": "You are an expert in generate json structure based on user query and schema."
                    },
                {
                    "role":    "user",
                    "content": PROMPT_TEMPLATE.format(query=example["query"][i], schema=example["schema"][i])
                    },
                {"role": "assistant", "content": example["response"][i]},
                ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=SFTConfig(
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=4,
                    warmup_steps=100,
                    max_steps=1000,
                    learning_rate=0.0002,
                    lr_scheduler_type="cosine",
                    eval_strategy="steps",
                    eval_steps=150,
                    weight_decay=0.01,
                    bf16=True,
                    logging_strategy="steps",
                    logging_steps=10,
                    output_dir="./" + OUTPUT_DIR,
                    optim="paged_adamw_8bit",
                    seed=42,
                    run_name=f"train-{OUTPUT_DIR}",
                    report_to="wandb",
                    save_steps=31,
                    save_total_limit=4),
            peft_config=LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=['o_proj', 'k_proj', 'q_proj', "v_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                    ),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            )

    # Fine-Tune the model
    trainer.train()

    # Optionally, you can save the fine-tuned LoRA adaptor:
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_checkpoint"))
    del model
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(OUTPUT_DIR, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
