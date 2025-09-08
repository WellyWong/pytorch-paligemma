# Based on Umar Jamil's implementation: https://github.com/hkproj/pytorch-paligemma
# Annotated with additional notes for educational purposes by Welly Wong
import os
import json
import glob
from safetensors import safe_open
from transformers import AutoTokenizer
from typing import Tuple
from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from processing import PaliGemmaProcessor

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Loads a PaliGemma model and its tokenizer from Hugging Face checkpoint.
    Reads the tokenizer, model configuration, and safetensors weights, initializes the model,
    loads the weights, and ties input/output embeddings before returning the model and tokenizer.
    """
    print(f'Loading tokenizer from {model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # Load safetensors_files one by one into the 'tensors' dictionary
    print(f'Loading {len(safetensors_files)} weight file(s) into memory...')
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using our class implementation (PaliGemmaForConditionalGeneration) and the config
    print('Loading model configuration')
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model using our 'tensors' dictionary
    print('Loading weights into model')
    model.load_state_dict(tensors, strict=False)

    # Share the weights of the input embedding layer with the output projection layer
    model.tie_weights()

    print('Model and tokenizer loaded successfully.')
    return (model, tokenizer)


def initialized_model(model_path: str, device: str = 'cuda'):
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    return model, tokenizer, processor