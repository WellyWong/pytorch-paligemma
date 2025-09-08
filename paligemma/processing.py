# Based on Umar Jamil's implementation: https://github.com/hkproj/pytorch-paligemma
# Annotated with additional notes for educational purposes by Welly Wong
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Union, Tuple, Iterable, Optional

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]   # IMAGENET_MEAN = [0.485, 0.456, 0.406], but hugging face doesn't use this
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]    # IMAGENET_STD = [0.229, 0.224, 0.225]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Prepares a prompt for PaliGemma by prepending image tokens and the beginning-of-sequence token.
    PaliGemma was trained using a newline character ('\n') to signal the end of the prompt,
    so this function appends it at the end. Note that although the original paper suggests
    tokenizing '\n' separately, the Hugging Face implementation does not do this.
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling=None,
           reducing_gap: Optional[int]=None) -> np.ndarray:
    """
    Resizes a PIL image to the specified (height, width) using optional resampling and reducing gap.
    """
    height, width = size
    resized_image = image.resize(   # .resize() is a method of the Pillow (PIL) Image class
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype=np.float32) -> np.ndarray:
    """
    Rescales image pixel values by a given factor and casts to the specified data type.
    """
    rescaled_image = image * scale     # rescale by factor: 1 / 255.0
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], 
              std: Union[float, Iterable[float]]) -> np.ndarray:
    """
    Normalizes image pixels by subtracting the mean and dividing by the standard deviation.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(images: List[Image.Image], 
                   size: Dict[str, int] = None, 
                   resample: Image.Resampling = None,
                   rescale_factor: float = None,
                   image_mean: Optional[Union[float, List[float]]] = None,
                   image_std: Optional[Union[float, List[float]]] = None) -> List[np.ndarray]:
    """
    Preprocessing pipeline for a list of images.
    Resizes, converts to numpy arrays, rescales, normalizes, and transposes to [channel, height, width] format.
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
 
    # Model expects images in: [channel, height, width], PIL Image: [height, width, channel]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    """
    Prepares text and images for the PaliGemma model. 
    Adds special tokens for images, object locations, and segmentation, 
    and converts input images to normalized tensors while tokenizing text prompts.
    """
    IMAGE_TOKEN = "<image>"  # Acts as a special placeholder token
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # For object detection, where each token can represent a discrete bounding box location.
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # For object segmentation, where each token can refer to a segmentation mask class or region.
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image],
                 padding: str = "longest", truncation: bool = False) -> dict:
        #assert len(images) == 1 and len(text) == 1, f'Received {len(images)} images for {len(text)} prompts.'
        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array with shape [batch_size, channels, height, width]
        pixel_values = np.stack(pixel_values, axis=0)

        # Convert numpy array to PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # ["{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"] for each prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        # Tokenizer returns input_ids (indices in the tokenizer's vocabulary) and attention_mask as PyTorch tensors
        inputs = self.tokenizer(input_strings, return_tensors='pt', padding=padding, truncation=truncation)
        return_data = {'pixel_values': pixel_values, **inputs}
        return return_data