import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration, Trainer, TrainingArguments

USE_QLORA = False

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main():
    ds = load_dataset('merve/vqav2-small', trust_remote_code=True)
    split_ds = ds["validation"].train_test_split(test_size=0.5)
    train_ds = split_ds["train"]
    processor = AutoProcessor.from_pretrained(
            model_id
            )
    lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            # nous entrainons seulment les linear layer du multi head attention
            # https://github.com/huggingface/transformers/blob/d3af76df58476830eb5b5981decc64af15e369f5/src/transformers/models/llama/modeling_llama.py#L254
            target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
            )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )

    model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            # you can use flash_attention_2 if you are on newer GPU than T4
            _attn_implementation="eager"
            )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print(model.get_nb_trainable_parameters())

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")]

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            if image.mode != 'RGB':
                image = image.convert('RGB')
            question = example["question"]
            answer = example["multiple_choice_answer"]
            messages = [
                {
                    "role":    "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                        ]
                    },
                {
                    "role":    "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                        ]
                    }
                ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    model_name = model_id.split("/")[-1]

    training_args = TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            learning_rate=1e-4,
            weight_decay=0.01,
            logging_steps=25,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=1,
            optim="paged_adamw_8bit",  # for 8-bit, keep this, else adamw_hf
            bf16=True,  # underlying precision for 8bit
            output_dir=f"./{model_name}-vqav2",
            hub_model_id=f"{model_name}-vqav2",
            report_to="tensorboard",
            remove_unused_columns=False,
            gradient_checkpointing=True
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=train_ds,
            )
    trainer.train()

if __name__ == "__main__":
    main()