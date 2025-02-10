import gc
import os
import argparse
import yaml

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, is_torch_xpu_available, is_torch_npu_available
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def clear_hardwares():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def main():
    # Parse command-line arguments to obtain the path to the config file.
    parser = argparse.ArgumentParser(description="Train model with hyperparameters in a config file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration file
    config = load_config(args.config)

    # ---------------------
    # Load Hyperparameters
    # ---------------------

    # Model checkpoint
    CHECKPOINT = config["checkpoint"]

    # Dataset configuration
    dataset_name = config.get("dataset_name", "ChristianAzinn/json-training")
    test_size = config.get("train_test_split", 0.2)
    random_seed = config.get("random_seed", 42)

    # Prompt templates
    RESPONSE_TEMPLATE = config["templates"]["response"]
    INSTRUCTION_TEMPLATE = config["templates"]["instruction"]
    PROMPT_TEMPLATE = config["templates"]["prompt"]

    # Load the dataset and create a train/test split
    ds = load_dataset(dataset_name)
    split_ds = ds["train"].train_test_split(test_size=test_size, seed=random_seed)
    train_dataset = split_ds["train"]
    test_dataset = split_ds["test"]

    # Load tokenizer and model from the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            device_map={"": PartialState().process_index},
            )

    # Data collator for training that uses the specified templates
    collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE,
            instruction_template=INSTRUCTION_TEMPLATE,
            tokenizer=tokenizer,
            mlm=False
            )

    # Define the function to format the prompts for each training example.
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

    # ---------------------------
    # Configure Training Settings
    # ---------------------------

    # Get SFT configuration from the config file
    sft_config_dict = config.get("sft_config", {})
    # If 'output_dir' is not provided, compute it based on the checkpoint name.
    if not sft_config_dict.get("output_dir"):
        OUTPUT_DIR = CHECKPOINT.split("/")[-1] + "-structure-output"
        sft_config_dict["output_dir"] = "./" + OUTPUT_DIR
    else:
        OUTPUT_DIR = sft_config_dict["output_dir"]

    # If 'run_name' is not provided, compute a default run name.
    if not sft_config_dict.get("run_name"):
        sft_config_dict["run_name"] = f"train-{OUTPUT_DIR}"

    # Create SFTConfig from the loaded configuration.
    sft_config = SFTConfig(**sft_config_dict)

    # Create LoRA configuration from the config file.
    lora_config = LoraConfig(**config.get("lora_config", {}))

    # Instantiate the trainer
    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            args=sft_config,
            peft_config=lora_config,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned LoRA adaptor
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.model.save_pretrained(final_checkpoint_dir)

    # Clean-up model and caches before merging.
    del model
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

    # Load the adapter, merge it with the base model, and save the merged model.
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16
            )
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
