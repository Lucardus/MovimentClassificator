import pathlib
pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gradio as gr
import numpy as np
import torch

learn = load_learner('model.pkl')

categorias = ('pessoa agachada', 'pessoa andando', 'pessoa correndo', 
              'pessoa deitada', 'pessoa parada', 'pessoa pulando', 'pessoa sentada')

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
    description="Envie uma foto e o modelo identifica o movimento da pessoa (Pulando, agachando, andando, correndo, parado, deitado ou sentado)"
)

intf.launch()
