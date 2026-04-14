import pathlib
pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gradio as gr
import numpy as np
import torch

learn = load_learner('model.pkl')

categorias = ('person crouching', 'person walking', 'person running', 
              'person lying down', 'person standing still', 'person jumping', 'person sitting')

def classificar(img):
    img_pil = Image.fromarray(np.uint8(img)).convert('RGB').resize((224, 224))
    img_tensor = torch.tensor(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        preds = learn.model(img_tensor)
        probs = torch.softmax(preds[0], dim=0)
    return dict(zip(categorias, map(float, probs)))

intf = gr.Interface(
    fn=classificar,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Classificador de Movimentos Humanos",
    description="Send a photo and the model that identifies the person's movement from among (Jumping, squatting, walking, running, standing, lying down or sitting)"
)

intf.launch()
