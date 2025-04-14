from diffusers import DiffusionPipeline
from PIL import Image
import requests
from io import BytesIO

def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True)

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png")
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline("Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", image=init_image).images[0]
#可视化
image.save("astronaut.png")