# Quick Start Guide

## Step 1: Edit Handler

Begin by updating the [handler.py](handler.py) file to meet your specific requirements. 

Subsequently, update the model weight with the one that you have uploaded.

````python
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model.eval()
````

## Step 2: Edit Dockerfile

Modify the model name `--model-name lang_detect` in the `torch-model-archiver` command as required.

## Step 3: Edit Configuration File

Edit the [config.properties](config.properties) file and replace `load_models=lang_detect.mar` with the name of the .mar file you created in the docker image.

## Step 4: Build Docker Image

Build the Docker image using the following command:

``docker build -t serve:latest .``

## Step 5: Run the Server

Execute the server with the command:

``docker run -p 8080:8080 serve:latest``

## Step 6: Test

Allow for the model to download onto the Docker container.

Once the download is complete, you can call the API with the following command:

````bash
curl --location 'http://localhost:8080/predictions/lang_detect' \
--header 'Content-Type: application/json' \
--data '[
    "Hello",
    "YO"
]'
````
