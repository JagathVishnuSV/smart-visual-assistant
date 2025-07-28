import torch
import clip
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def answer_question(image_np, question):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    # Generate candidate answers (can be enhanced)
    candidates = ["person", "cup", "phone", "laptop", "nothing", "book", "pen", "bottle", "table", "mouse"]

    # Create prompts
    texts = [f"A photo of a {item}" for item in candidates]
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        logits_per_image, _ = model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best_idx = int(np.argmax(probs))
    best_answer = candidates[best_idx]
    return image_np, f"The assistant thinks it's: {best_answer}"
