# gemma-captioner: Quick and easy video and image captioning with Gemma 3

These are one shot scripts designed to work with Gemma 3 by Google that automatically performs captioning on an entire folder of images or videos, and will create caption .txt files with the same file name as the corresponding image or video. Best used in a fine-tuning pipeline for open source video packages like Mochi, Wan, LTX, or Hunyuan Video (the latter three best accomplished through diffusion-pipe.)

You can edit the prompt at the top to have it provide whatever kind of captions you prefer, though most video packages do best with a single descriptive line.

Image captioning works extremely well. Video captioning works by taking a certain number of keyframes over the course of the video, which you can edit at the top. More keyframes examine motion more accurately, but it is possible to overload the attention mechanism and get an error if too many keyframes are provided. What you can get away with varies from video to video. Be advised that as of now most open-source packages only support between 80-250 frames, max, for video generation and fine-tuning, so this works best on a group of small curated clips rather than a single large unedited video. 

## Image captioning

Usage:
```
python gemma3-image-captioning.py --image_folder images/ --hf_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Iterates over an folder of images and provides a caption. hf_token optional, but required if you are trying to download a gated model.

Edit the arguments at the top of the .py to your liking:
```
# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-27b-pt"

# Prompt for image captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a short, single-line description of this image for training data."
# =====================================
```

## Video captioning

Usage:
```
python gemma3-video-captioning.py --videos_folder videos/ --hf_token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Iterates over an folder of videos and provides a caption. hf_token optional, but required if you are trying to download a gated model.

Edit the arguments to your liking:
```
# ===== USER MODIFIABLE SETTINGS =====
# Model to use (e.g., google/gemma-3-4b-it, google/gemma-3-12b-it, google/gemma-3-27b-it)
MODEL_ID = "unsloth/gemma-3-12b-it"

# Prompt for video captioning - modify this to change what kind of captions you get
CAPTION_PROMPT = "Provide a detailed, thorough caption of movement you see in the video. Only describe movement as this will be very important during the training process."

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
```
