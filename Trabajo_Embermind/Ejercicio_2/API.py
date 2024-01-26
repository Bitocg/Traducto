from fastapi import FastAPI
from pydantic import BaseModel
import torch
from keras_transformer import get_model, decode
import uvicorn

app = FastAPI()

#Ruta del mode
filename = "modelo_pesos_GT.h5"

#Cargamos los diccionarios usados en el entrenamiento del modelo
source_token_dict = torch.load("Dic/source_token_dict.pth")
target_token_dict = torch.load("Dic/target_token_dict.pth")
target_token_dict_inv = torch.load("Dic/target_token_dict_inv.pth")

#Cargamos el modelo
model = get_model(
    token_num = max(len(source_token_dict),len(target_token_dict)), #Le pasamos cuántos tokens hay en el dic con más tokens
    embed_dim = 32,
    encoder_num = 4,  #nº de codificadores
    decoder_num = 4,  #nº de decodificadores
    head_num = 8,     #nº de bloques atencionales
    hidden_dim = 128, #nº de neuronas ocultas
    dropout_rate = 0.05, #% de neuronas que desactivamos (overfitting)
    use_same_embed = False, #Utiliza capas de embedding separadas para las entradas de origen y destino.
)

#Le pasamos los pesos al modelo
model.load_weights(filename)

#Función de traducción
def translate_sentence(sentence):
    sentence_tokens = [tokens + ["<END>", "<PAD>"] for tokens in [sentence.split(" ")]] #Agregamos el relleno y el token del final
    tr_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in sentence_tokens][0] #Codificamos la entrada

    #LLamamos a la función decode de la librería
    decoded = decode(
      model,#Nuestro modelo
      tr_input,#Lo que tiene que traducir
      start_token = target_token_dict["<START>"],
      end_token = target_token_dict["<END>"],
      pad_token = target_token_dict["<PAD>"]
    )
    translated_text = 'Traducción: {}'.format(' '.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1])))
    return translated_text


#Define la estructura de las solicitudes que la API espera recibir.
class TranslationRequest(BaseModel):
    sentence: str

#Indica que translate_text maneja solicitudes POST en esa ruta
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    #Llama a nuestra función de inferencia con la frase a traducir
    translated_text = translate_sentence(request.sentence)

    #Devuelve la frase traducida
    return {"translated_text": translated_text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)