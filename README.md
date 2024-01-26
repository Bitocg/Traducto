# Traductor
## Carpeta Ejercicio_1: 
Hay dos tipos de modelos bastante similares con la única diferencia de que el 1º es con un corpus limpio de pares de frases en inglés y español y el 2º es con un corpus más 
"sucio" que es la wikipedia en español y en inglés. Mi idea era presentaros solo el segundo pero al intentar implementarlo en el ejercicio 2, me di cuenta que necesitaba los diccionarios de tokens con los que entrenó el modelo (los cuales no guarde durante el entrenamiento) y al haber seleccionado aleatoriamente 200.000 pares de frases no pude replicar la aletoriedad para conseguir los diccionarios con los que entrenó y tampoco me daba tiempo a volver a entrenarlo, entonces tuve que usar el modelo 1 para el ejercicio 2.

## Carpeta Ejercicio_2: 
Implementación del modelo 1 con Fast Api. Esta carpeta contiene los archivos Dic (contiene los dic con los que entrenó el modelo), el script API.py y el archivo de los pesos del modelo.

## Carpeta Ejercicio_3: 
Contiene un notebook donde selecciono un modelo de Hugging Face y un script que implementa un flujo de ejecución asíncrono que, mediante la API de FastAPI ejecuta los dos modelos (El del paso 1 y el del paso 3) y muestra ambos resultados.

## SCript Request:
script para realizar la petición a la API (vale para ambas APIS)

Los notebooks están programados para ejecutarse en google colab y los scripts disponen de su enviroment para ser ejecutados.
