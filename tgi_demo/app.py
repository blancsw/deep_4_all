"""
Do a document classification based on this labels: form, id, bill, resume, other
In this format:
label:
"""
import base64
import mimetypes

import gradio as gr
from openai import OpenAI
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")


def image_to_base64(image_path):
    # Detect the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Could not determine the MIME type of the image.")

    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read()).decode('utf-8')

    # Return the base64-encoded string prefixed with the appropriate MIME type
    return f"data:{mime_type};base64,{encoded_string}"


# Chat function updated to take additional parameters
def chat_with_llm(message, history, temperature, top_p, presence_penalty, frequency_penalty):
    messages = []
    files = []

    for couple in history:
        if type(couple["content"]) is tuple:
            mime_type, _ = mimetypes.guess_type(couple["content"][0])
            if mime_type and mime_type.startswith('image/'):
                messages.append({
                    "role":    "user",
                    "content": [{
                        "type":      "image_url",
                        "image_url": {
                            "url": image_to_base64(couple["content"][0])
                            }
                        }]
                    })
        elif isinstance(couple["content"], str):
            messages.append({"role": couple["role"], "content": couple["content"]})

    for file in message["files"]:
        mime_type, _ = mimetypes.guess_type(file)
        if mime_type and mime_type.startswith('image/'):
            files.append({
                "type":      "image_url",
                "image_url": {"url": image_to_base64(file)}
                })
    if len(files) > 0:
        messages.append({"role": "user", "content": [{"type": "text", "text": message["text"]}] + files})
    else:
        messages.append({"role": "user", "content": message["text"]})

    chat_completion = client.chat.completions.create(
            model="HuggingFaceTB/SmolVLM-256M-Instruct",
            stream=True,
            messages=messages,
            max_tokens=1024
            )
    partial_message = ""
    for token in chat_completion:
        content = token.choices[0].delta.content
        if token.choices[0].finish_reason is not None:
            break
        partial_message += content
        yield partial_message


chat_interface = gr.ChatInterface(
        fn=chat_with_llm,
        multimodal=True,
        type='messages',
        examples=[
            {
                "text":  "Classify this document from this labels: bill, form, id, resume\nOnly return the label in this format\n\nlabel:",
                "files": ["bill.jpg"]
                },
            {
                "text":  "Classify this document from this labels: bill, form, id, resume\nOnly return the label in this format\n\nlabel:",
                "files": ["resume.jpg"]
                },
            {
                "text":  "Expliquer les principales différences des cartes graphiques",
                "files": ["2.png"]
                }
            ],
        )

chat_interface.launch()

