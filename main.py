#from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel): #defino el modelo de la entrada por body
    texto: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/enviar/")
async def create_item(item: Item):
    clasificacion, score = ClasificarTexto(item.texto)
    return {"clasificacion": clasificacion , "score": round(float(score),3)}

#texto = "Un coche inicia un viaje de 495 Km. a las ocho y media de la mañana con una velocidad media de 90 Km/h"
#texto = "Un caracol recorre en línea recta una distancia de 10,8 m en 1,5 h. ¿Qué distancia recorrerá en 5 min?"
#texto = "Calcula la energía potencial que posee un libro de 500 gramos de masa que está colocado sobre una mesa de 80 centímetros de altura."
#texto = "Un correa de cuero esta enrollada en una polea a 20 cm de diámetro. Se aplica a la correa una fuerza de 60 N. ¿Cuál es el momento de torsión en el centro del eje?"
#texto = "Desde un edificio se deja caer una pelota, que tarda 8 segundos en llegar al piso. ¿con que velocidad impacta la pelota contra el piso?"

def ClasificarTexto(texto):
    classifier = pipeline("text-classification", 
                        model="dracero/autotrain-dracero-fine-tuned-physics-2123168626")

    output = classifier(
        texto
    )
    #return float(output[0]['score'])
    if float(output[0]['score'])<0.6:
        return [False,output[0]['score']]
    else:
        return [output[0]['label'],output[0]['score']]
