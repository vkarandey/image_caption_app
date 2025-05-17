from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import os
import gdown


app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


text_config = {
    "attention_probs_dropout_prob": 0.0,
    "encoder_hidden_size": 768,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "hidden_size": 512,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "label_smoothing": 0.0,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "blip_text_model",
    "num_attention_heads": 8,
    "num_hidden_layers": 6,
    "projection_dim": 768,
    "use_cache": True,
    "vocab_size": 30524,
}

vision_config = {
    "attention_dropout": 0.0,
    "dropout": 0.0,
    "hidden_act": "gelu",
    "hidden_size": 768,
    "image_size": 384,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "model_type": "blip_vision_model",
    "num_attention_heads": 12,
    "num_channels": 3,
    "num_hidden_layers": 6,
    "patch_size": 16,
    "projection_dim": 512,
}

config = BlipConfig(
    image_text_hidden_size=256,
    initializer_factor=1.0,
    initializer_range=0.02,
    label_smoothing=0.0,
    logit_scale_init_value=2.6592,
    projection_dim=512,
    text_config=text_config,
    vision_config=vision_config,
)

def download_weights():
    url = "https://drive.google.com/uc?id=1w7hY_dpYc-QJ_qUzBkz-2uBqxnfS0lko" 
    weights_path = "student_epoch6.pt"
    if not os.path.exists(weights_path):
        print("Скачиваем веса модели...")
        gdown.download(url, weights_path, quiet=False)
    else:
        print("Веса модели уже загружены.")

download_weights()

model = BlipForConditionalGeneration(config)
model.load_state_dict(torch.load("student_epoch6.pt", map_location=device), strict=False)
model.to(device)
model.eval()

def generate_caption(image_pil):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=32, num_beams=5)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded.jpg")
            file.save(image_path)

            image = Image.open(image_path).convert("RGB")
            caption = generate_caption(image)

    return render_template("index.html", caption=caption, image_path=image_path)

if __name__ == "__main__":
    print("Flask запускается...")
    app.run(debug=True)