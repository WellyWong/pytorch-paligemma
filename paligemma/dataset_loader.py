import os
import glob
from datasets import Dataset, concatenate_datasets

def load_flickr30k():
    arrow_path = r"C:\Users\welly\.cache\huggingface\datasets\nlphuji___flickr30k\TEST\1.1.0\2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
    print('Loading flickr30k dataset ...')
    # Find all shard files (9 arrow files)
    shards = sorted(glob.glob(os.path.join(arrow_path, "flickr30k-test-*.arrow")))
    print(f'Found {len(shards)} shards')

    # Load and concatenate all shards into a single Dataset
    datasets_list = [Dataset.from_file(shard) for shard in shards]
    dataset = concatenate_datasets(datasets_list)
    print('Loaded all shards')
    
    print('Loading images, decoding from Arrow file -> PIL.Image object (this may take a while) ...')
    images = list(dataset['image'])
    captions = [caps[0] for caps in dataset['caption']]
    caption_to_image = list(range(len(dataset["caption"])))
    return images, captions, caption_to_image