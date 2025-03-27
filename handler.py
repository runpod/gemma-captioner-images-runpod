import os
import json
import base64
import tempfile
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# Model settings
MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gemma-3-27b-pt")
CAPTION_PROMPT = os.environ.get("CAPTION_PROMPT", "Provide a short, single-line description of this image for training data.")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "true").lower() == "true"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Initialize model and processor
model = None
processor = None

def setup_model():
    global model, processor
    
    print(f"Loading model: {MODEL_ID}")
    
    # Configure token if provided
    token_param = {"token": HF_TOKEN} if HF_TOKEN else {}
    
    # Configure quantization if requested
    if USE_QUANTIZATION:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
    
    # Load the model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
        **token_param,
    ).eval()
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, **token_param)
    
    print(f"Model loaded on {device}")

def caption_image(image, prompt=CAPTION_PROMPT, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a caption for the given image."""
    global model, processor
    
    # Initialize model if not already loaded
    if model is None or processor is None:
        setup_model()
    
    # Create messages for the model with custom prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Track input length to extract only new tokens
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate caption
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    
    # Extract only the newly generated tokens
    generated_tokens = outputs[0][input_len:]
    
    # Decode the caption
    caption = processor.decode(generated_tokens, skip_special_tokens=True)
    
    # Ensure caption is a single line
    caption = caption.replace('\n', ' ').strip()
    return caption

def handler(event):
    try:
        body = json.loads(event["body"])
        
        # Get parameters from request or use defaults
        prompt = body.get("prompt", CAPTION_PROMPT)
        max_new_tokens = int(body.get("max_new_tokens", MAX_NEW_TOKENS))
        
        # Extract image data (base64 encoded)
        image_data = body.get("image")
        if not image_data:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No image data provided"})
            }
        
        # Check if it's a URL or base64
        if image_data.startswith('http'):
            import requests
            response = requests.get(image_data)
            if response.status_code != 200:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Failed to download image from URL: {response.status_code}"})
                }
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes).convert("RGB")
        else:
            # Decode base64 image
            if image_data.startswith('data:image'):
                # Handle data URI scheme
                image_data = image_data.split(',')[1]
            image_bytes = BytesIO(base64.b64decode(image_data))
            image = Image.open(image_bytes).convert("RGB")
        
        # Generate caption
        caption = caption_image(image, prompt, max_new_tokens)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "caption": caption
            })
        }
    
    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }

# For local testing
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
