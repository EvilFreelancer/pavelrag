import ollama

# DEFAULT_MODEL_NAME = "siasi/qwen2-vl-7b-instruct"
DEFAULT_MODEL_NAME = "llama3.2-vision:11b"


def describe_image(image_path: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    response = ollama.chat(
        model=model_name,
        stream=False,
        messages=[{
            'role': 'user',
            'content': 'Write all text from document. Document text:',
            'images': [image_path]
        }]
    )
    return response['message']['content']
