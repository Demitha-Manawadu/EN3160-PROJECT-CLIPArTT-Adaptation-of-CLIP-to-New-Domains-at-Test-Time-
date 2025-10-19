import os
import sys
from PIL import Image
import torch

# Make "clip" importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from clip import load, tokenize

def main(image_path=None, model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, preprocess = load(model_name, device=device)

    if image_path and os.path.isfile(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (512, 512), color=(200, 200, 200))

    x = preprocess(img).unsqueeze(0).to(device)
    text = tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

    with torch.no_grad():
        z_img = model.encode_image(x)
        z_txt = model.encode_text(text)
        z_img = z_img / z_img.norm(dim=1, keepdim=True)
        z_txt = z_txt / z_txt.norm(dim=1, keepdim=True)
        logits = z_img @ z_txt.T
        probs = logits.softmax(dim=-1).cpu().numpy()

    print("Class order: ['cat', 'dog']")
    print("Probabilities:", probs.tolist())

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(img_path)
