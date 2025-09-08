import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import PreTrainedTokenizer
from typing import List, Tuple
from sklearn.preprocessing import normalize
from dataset_loader import load_flickr30k
from utils import initialized_model
from embedding_extractor import extract_image_embeddings, extract_text_embeddings
from gemma import PaliGemmaForConditionalGeneration

def retrieve_top_images_from_text(
    query_text: str,
    model: PaliGemmaForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    img_embs_np: np.ndarray,
    top_k: int = 10,
    batch_size: int = 1,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k most similar images for a given text query in a zero-shot setting.
    """
    # Compute text embedding
    text_embs = extract_text_embeddings(model, tokenizer, [query_text], batch_size=batch_size, device=device)
    
    # Convert to numpy and normalize the query text
    text_embs_np = normalize(text_embs.cpu().numpy(), axis=1)
    
    # Get the 1-D vector (1, 2048) -> (2048,)
    query_vec = text_embs_np[0]
    
    # Compute top image indices and similarity scores
    sims = img_embs_np @ query_vec
    top_indices = np.argsort(-sims)[:top_k]
    scores = sims[top_indices]   
    return top_indices, scores


def show_top_images(images, top_img_indices, display_top_1=False, save_path=None, title=None, n_rows=2):
    if display_top_1:
        idx = top_img_indices[0]  # rank 1 image
        plt.figure(figsize=(4, 4))
        plt.imshow(images[idx])
        plt.axis('off')   
        plt.title('Rank 1')
        plt.show()
    else:
        top_k = len(top_img_indices)
        n_cols = math.ceil(top_k / n_rows)
        
        plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        for i, idx in enumerate(top_img_indices):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(images[idx])
            plt.axis('off')
            plt.title(f'Rank {i + 1}')
        if title:
            plt.suptitle(f'Input prompt: "{title}"', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved retrieved images to {save_path}")
        plt.show(block=True)  # keeps the window open until you close it


def main():
    """
    Main function to load Flickr30k test data, initialize the model,
    extract or load image and text embeddings, and optionally save them.
    """
    model_path: str = r"C:/Users/welly/Documents/Course_DataMining_Project/models/paligemma-3b-pt-224"
    device: str = "cuda"
    save_path: str = "flickr30k_img_embs.pt"
    batch_size: int = 32

    model, tokenizer, processor = initialized_model(model_path, device)

    images, _, _ = load_flickr30k()

    # Check if embeddings already exist
    if os.path.exists(save_path):
        print(f'Embeddings file {save_path} exists. Loading embeddings...')
        data = torch.load(save_path, weights_only=True)
        img_embs: torch.Tensor = data["img_embs"]
        # txt_embs: torch.Tensor = data["txt_embs"]
        # caption_to_image: List[int] = data["caption_to_image"]
    else:
        print('Extracting embeddings... This may take a while.')

        img_embs: torch.Tensor = extract_image_embeddings(
            images, processor, model, device=device, batch_size=batch_size
        )

        # txt_embs: torch.Tensor = extract_text_embeddings(
        #     model, tokenizer, captions, batch_size=batch_size, device=device
        # )

        torch.save({
            "img_embs": img_embs,
            # "txt_embs": text_embs,
            # "caption_to_image": caption_to_image
        }, save_path)
        print(f'Embeddings saved to {save_path}')

    # Print shapes for sanity check
    print(f'Image embeddings shape: {img_embs.shape}')
    # print(f'Text embeddings shape: {text_embs.shape}')
    print(f'Number of images: {len(images)}')

    img_embs_np = normalize(img_embs.cpu().numpy())
    # txt_embs_np = normalize(txt_embs.cpu().numpy())

    # Example retrieval
    query_text = "A dog playing with a ball"
    top_img_indices, _ = retrieve_top_images_from_text(query_text, model, tokenizer, img_embs_np)
    show_top_images(images, top_img_indices, display_top_1=False, save_path=None, title=query_text)


if __name__ == "__main__":
    main()
