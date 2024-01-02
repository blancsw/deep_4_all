# Install

````bash
pip install git+https://github.com/LAION-AI/Open-Assistant.git#subdirectory=oasst-data
````

# Usage and format

Format the dataset ino [chat template](https://huggingface.co/docs/transformers/chat_templating)

````python
from datasets import load_dataset

dataset = load_dataset("blancsw/oasst2_top1_chat_format")
````