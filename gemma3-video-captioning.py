# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-12b-it"

# Prompt for video captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a detailed, thorough caption of movement you see in the video. Only describe movement as this will be very important during the training process"

# Number of frames to extract from video for captioning.
# First, it calculates the total number of frames in the video by getting the frame count property from the video file.
# Then, it uses one of two methods depending on how many frames are in the video:
# If the video has fewer frames than the requested number (set by NUM_FRAMES at the top or with the --num_frames parameter), it will use all available frames.
# If the video has more frames than requested, it selects frames at regular intervals.

# There is a tradeoff involved here. Increasing NUM_FRAMES will give the captioning process more keyframes to work with.
# However, increasing NUM_FRAMES too high will cause attention mechanism errors.
# Where this point occurs can vary even from video to video. However, using larger Gemma models seems to give the process more breathing room to work with. 

NUM_FRAMES = 12

# Maximum new tokens to generate for caption
MAX_NEW_TOKENS = 100
# =====================================

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gemma 3 Video Captioning')
    parser.add_argument('--video_folder', type=str, required=True, 
                        help='Path to folder containing videos')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Path to folder for saving caption files (defaults to video folder)')
    parser.add_argument('--model_id', type=str, default=MODEL_ID, 
                        help=f'Gemma 3 model ID (default: {MODEL_ID})')
    parser.add_argument('--prompt', type=str, default=CAPTION_PROMPT,
                        help='Prompt for video captioning')
    parser.add_argument('--num_frames', type=int, default=NUM_FRAMES,
                        help=f'Number of frames to extract (default: {NUM_FRAMES})')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS, 
                        help=f'Maximum number of tokens to generate (default: {MAX_NEW_TOKENS})')
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


def extract_frames(video_path, num_frames):
    """Extract frames from a video file."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        # Calculate frame indices to extract
        if total_frames <= num_frames:
            # If video has fewer frames than requested, use all frames
            frame_indices = list(range(total_frames))
        else:
            # Otherwise, extract frames at regular intervals
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        # Extract the frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            else:
                print(f"Warning: Failed to read frame {idx}")
        
        # Release the video capture object
        cap.release()
        
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error extracting frames: {str(e)}\n{traceback_str}")
        return []


def caption_video(video_path, model, processor, prompt, num_frames, max_new_tokens):
    """Generate a caption for the given video."""
    try:
        # Extract frames from video
        frames = extract_frames(video_path, num_frames)
        if not frames:
            return f"Error processing {os.path.basename(video_path)}: Failed to extract any frames"
        
        # Create messages for the model with frames
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Add all frames to the message content
        for frame in frames:
            messages[0]["content"].append({"type": "image", "image": frame})
        
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
        return f"Error processing {os.path.basename(video_path)}: {str(e)}\n{traceback_str}"


def main():
    """Main function to run the video captioning pipeline."""
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
    
    # Set output folder
    output_folder = args.output_folder if args.output_folder else args.video_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up the model
    model, processor = setup_model(
        args.model_id, args.hf_token, args.quantize
    )
    
    # Get a list of video files
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = [
        os.path.join(args.video_folder, f) for f in os.listdir(args.video_folder)
        if os.path.splitext(f.lower())[1] in supported_formats
    ]
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Using prompt: '{args.prompt}'")
    print(f"Using {args.num_frames} frames per video")
    
    # Process each video and save caption to individual text file
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        video_base = os.path.splitext(video_name)[0]
        output_path = os.path.join(output_folder, f"{video_base}.txt")
        
        print(f"[{i}/{len(video_files)}] Processing {video_name}")
        
        caption = caption_video(
            video_path, model, processor, args.prompt, args.num_frames, args.max_new_tokens
        )
        
        # Save caption to a text file with the same name as the video
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(caption)
        
        print(f"Caption saved to {output_path}")
    
    print("All captions saved successfully")


if __name__ == "__main__":
    main()
