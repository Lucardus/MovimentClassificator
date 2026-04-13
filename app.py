from fastai.vision.all import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gradio as gr

learn = load_learner(r'C:\Users\lucas\MovimentClassificator\acoes\model.pkl')

categorias = ('pessoa agachada', 'pessoa andando', 'pessoa correndo', 
              'pessoa deitada', 'pessoa parada', 'pessoa pulando', 'pessoa sentada')

def classificar(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categorias, map(float, probs)))

intf = gr.Interface(
    fn=classificar,
    inputs=gr.Image(width=224, height=224),
    outputs=gr.Label(num_top_classes=3),
    title="Classificador de Movimentos Humanos",
    description="Envie uma foto e o modelo identifica o movimento da pessoa" 
)

intf.launch() 