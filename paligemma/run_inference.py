import torch
from utils import initialized_model
from inference import test_inference

def main():
    model_path = r"C:/Users/welly/Documents/Course_DataMining_Project/models/paligemma-3b-pt-224"
    device = "cuda"
    prompt = "Why is the child crying"
    image_path = r"assets/test_image.jpg"
    num_runs = 5

    model, _, processor = initialized_model(model_path, device)

    print('Running inference ...')
    with torch.no_grad():
        for i in range(num_runs):
            print(f"\n=== Run {i+1} ===")
            test_inference(
                model, 
                processor, 
                device, 
                prompt, 
                image_path, 
                max_tokens_to_generate=100, 
                temperature=1.1,  # higher temperature -> more randomness
                top_p=0.95,
                do_sample=True
            )
        
if __name__ == "__main__":
    main()