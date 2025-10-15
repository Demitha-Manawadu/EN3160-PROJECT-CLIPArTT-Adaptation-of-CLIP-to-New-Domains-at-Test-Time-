from PIL import Image
import torch, os
import sys

# Allow "from third_party.clip import load, tokenize"
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from third_party.clip import load, tokenize

def main(image_path=None, model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Available models:", load.__globals__["available_models"]())

    model, preprocess = load(model_name, device=device)

    # Use a provided image if you have one; otherwise make a dummy gray image
    if image_path and os.path.isfile(image_path):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (512, 512), color=(200, 200, 200))

    img_t = preprocess(img).unsqueeze(0).to(device)
    text = tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img_t)         # [1, D]
        text_features  = model.encode_text(text)           # [2, D]

        # normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=1, keepdim=True)

        # similarity and probabilities
        logits = image_features @ text_features.T          # [1, 2]
        probs  = logits.softmax(dim=-1).cpu().numpy()

    print("Class order: ['cat','dog']")
    print("Probabilities:", probs.tolist())

if __name__ == "__main__":
    # Optional: pass a path to an image file
    # python scripts/clip_sanity.py /path/to/image.jpg
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(img_path)
