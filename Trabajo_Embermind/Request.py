import requests

url = "http://127.0.0.1:8000/translate"
sentence_to_translate = "hola buenos dias"

#Hacemos la request a la api y nos devuelve un json con la traducción
response = requests.post(url, json={"sentence": sentence_to_translate})
translated_text = response.json()["translated_text"]

print("Traducción:", translated_text)