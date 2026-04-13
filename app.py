import pathlib
pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gradio as gr
import numpy as np

learn = load_learner('model.pkl')

categorias = ('pessoa agachada', 'pessoa andando', 'pessoa correndo', 
              'pessoa deitada', 'pessoa parada', 'pessoa pulando', 'pessoa sentada')

def classificar(img):
    img_pil = Image.fromarray(np.uint8(img))
    img_pil.save('/tmp/temp_img.jpg')
    img_fastai = PILImage.create('/tmp/temp_img.jpg')
    pred, idx, probs = learn.predict(img_fastai)
    return dict(zip(categorias, map(float, probs)))

intf = gr.Interface(
    fn=classificar,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="Classificador de Movimentos Humanos",
    description="Envie uma foto e o modelo identifica o movimento da pessoa"
)

intf.launch()
