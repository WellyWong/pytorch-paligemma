import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer
from processing import PaliGemmaProcessor
from gemma import PaliGemmaForConditionalGeneration
from typing import List
from PIL import Image

def extract_image_embeddings(
    images: List[Image.Image],
    processor: PaliGemmaProcessor,
    model: PaliGemmaForConditionalGeneration,
    device: str = 'cuda',
    batch_size: int = 32
) -> torch.Tensor:
    img_embs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = [img.convert('RGB') for img in images[i:i+batch_size]]
            inputs = processor(
                text=[""] * len(batch_imgs),
                images=batch_imgs,
                padding='longest',
                truncation=False
            )
            pixel_values = inputs['pixel_values'].to(device)
            vision_out = model.vision_tower(pixel_values)
            img_proj = model.multi_modal_projector(vision_out)
            img_pooled = img_proj.mean(dim=1)
            img_embs.append(img_pooled.detach().cpu())
    img_embs = torch.cat(img_embs, dim=0)
    return img_embs


def extract_text_embeddings(
    model: PaliGemmaForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    captions: List[str],
    batch_size: int = 32,
    device: str = 'cuda'
) -> torch.Tensor:
    pad_token_id = model.pad_token_id
    image_token_id = model.config.image_token_index
    all_txt_embs = []
    for i in range(0, len(captions), batch_size):
        batch_caps = captions[i:i+batch_size]
        tokenized = tokenizer(
            batch_caps,
            return_tensors='pt',
            padding='longest',
            truncation=False,
        ).to(device)
        input_ids = tokenized['input_ids']             # [batch_size, seq_len]
        attention_mask = tokenized['attention_mask']   # [batch_size, seq_len]
        with torch.no_grad():
            token_embs = model.language_model.get_input_embeddings()(input_ids)   # [batch_size, seq_len, hidden_dim]
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
            q_len = input_ids.shape[1]
            causal_mask = torch.zeros((input_ids.shape[0], 1, q_len, q_len), dtype=token_embs.dtype, device=device)            
            hidden_states = model.language_model.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                inputs_embeds=token_embs,
                kv_cache=None
            )
            valid_mask = (input_ids != pad_token_id) & (input_ids != image_token_id)
            # pooled = []
            # for b in range(hidden_states.size(0)):    # TODO: too slow, Vectorized!
            #     valid = hidden_states[b][valid_mask[b]]
            #     pooled.append(valid.mean(dim=0) if valid.numel() else torch.zeros(hidden_states.size(-1)))
            # pooled = torch.stack(pooled)    # [batch_size, hidden_dim]
            mask_float = valid_mask.float()   # [batch_size, seq_len]
            
            # Avoid division by zero for mean
            mask_sum = mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # [batch_size, 1]
            
            # Expand for hidden_dim
            mask_expanded = mask_float.unsqueeze(-1)          # [batch_size, seq_len, 1]
            
            # Apply mask to hidden_states
            masked_hidden = hidden_states * mask_expanded     # zeros out invalid tokens
            
            # Mean pooling
            mean_pool = masked_hidden.sum(dim=1) / mask_sum   # [batch_size, hidden_dim]
            
        all_txt_embs.append(mean_pool.detach().cpu())
    return torch.cat(all_txt_embs, dim=0)