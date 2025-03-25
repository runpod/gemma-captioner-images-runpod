# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-27b-pt"

# Prompt for image captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a short, single-line description of this image for training data."
# =====================================

import os
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gemma 3 Image Captioning')
    parser.add_argument('--image_folder', type=str, required=True, 
                        help='Path to folder containing images')
    parser.add_argument('--model_id', type=str, default=MODEL_ID, 
                        help=f'Gemma 3 model ID (default: {MODEL_ID})')
    parser.add_argument('--prompt', type=str, default=CAPTION_PROMPT,
                        help='Prompt for image captioning')
    parser.add_argument('--max_new_tokens', type=int, default=256, 
                        help='Maximum number of tokens to generate')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='Hugging Face API token (optional, required for gated models)')
    parser.add_argument('--quantize', action='store_true', 
                        help='Use 8-bit quantization to reduce memory usage')
    return parser.parse_args()


def setup_model(model_id, hf_token, use_quantization):
    """Set up the Gemma 3 model and processor."""
    print(f"Loading model: {model_id}")
    
    # Configure token if provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        token_param = {"token": hf_token}
        print("Using provided Hugging Face token")
    else:
        token_param = {}
        print("No Hugging Face token provided (this will only work for non-gated models)")
    
    # Configure quantization if requested
    if use_quantization:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None
    
    # Load the model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            **token_param,
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_id, **token_param)
        
        print(f"Model loaded on {device}")
        return model, processor
    except Exception as e:
        if not hf_token:
            print("ERROR: Failed to load model. This may be a gated model that requires a token.")
            print("Try running again with --hf_token parameter.")
        raise e


def caption_image(image_path, model, processor, prompt, max_new_tokens):
    """Generate a caption for the given image."""
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
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
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"Error processing {os.path.basename(image_path)}: {str(e)}\n{traceback_str}"


def main():
    """Main function to run the image captioning pipeline."""
    args = parse_arguments()
    
    # Check if transformers version supports Gemma 3
    try:
        import transformers
        version = transformers.__version__
        print(f"Using transformers version: {version}")
        if "4.50" not in version and "4.5" not in version:
            print("Warning: Gemma 3 requires transformers version 4.50.0 or newer.")
            print("You may need to install it with: pip install git+https://github.com/huggingface/transformers@v4.50.0-Gemma-3")
    except ImportError:
        print("Warning: Could not detect transformers version.")
    
    # Set up the model
    model, processor = setup_model(
        args.model_id, args.hf_token, args.quantize
    )
    
    # Get a list of image files
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [
        os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)
        if os.path.splitext(f.lower())[1] in supported_formats
    ]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using prompt: '{args.prompt}'")
    
    # Process each image and save caption to individual text file
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        image_base = os.path.splitext(image_name)[0]
        output_path = os.path.join(args.image_folder, f"{image_base}.txt")
        
        print(f"[{i}/{len(image_files)}] Processing {image_name}")
        
        caption = caption_image(
            image_path, model, processor, args.prompt, args.max_new_tokens
        )
        
        # Save caption to a text file with the same name as the image
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(caption)
        
        print(f"Caption saved to {output_path}")
    
    print("All captions saved successfully")


if __name__ == "__main__":
    main()
