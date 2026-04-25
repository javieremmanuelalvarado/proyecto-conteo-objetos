\- Conteo de Objetos en Imágenes

\-Procesamiento de Imágenes Biomédicas



Proyecto desarrollado en Python para la segmentación, análisis y conteo de objetos en imágenes utilizando técnicas fundamentales implementadas manualmente.







\##  Descripción



Este proyecto implementa un pipeline completo de procesamiento de imágenes que incluye:



\- Conversión de RGB a escala de grises

\- Filtrado espacial (convolución manual)

\- Transformaciones de intensidad (negativo)

\- Detección de bordes (Sobel)

\- Umbralización

\- Segmentación por componentes conectados

\- Eliminación de objetos pequeños (ruido)

\- Cálculo de propiedades de objetos (área, centroides)

\- Detección de esquinas (Harris)

\- Conteo final de objetos



El enfoque principal es \*\*no depender de funciones automáticas\*\*, sino implementar los algoritmos desde cero para entender su funcionamiento.







\##  Tecnologías utilizadas



\- Python

\- NumPy

\- Matplotlib

\- OpenCV (uso mínimo)







\##  Estructura del proyecto

proyecto-conteo-objetos/

&#x20;contador\_objetos/

&#x09;preprocesamiento.py

&#x09;segmentacion.py 

&#x09;init.py



&#x20;ejemplos/

&#x09;demo.py



&#x20;imagenes\_prueba/

&#x09;(imágenes de prueba)



&#x20;requirements.txt

&#x20;setup.py

&#x20;README.md



\##  Cómo usar el proyecto



\### 1. Clonar el repositorio



git clone https://github.com/javieremmanuelalvarado/proyecto-conteo-objetos.git

cd proyecto-conteo-objetos



\### 2. Instalar dependencias

pip install -r requirements.txt



\### 3. Instalar el proyecto como librería

pip install -e .



\### 4. Ejecutar el ejemplo

\-python ejemplos/demo.py



\### 5. Usarlo en otro script

Después de instalarlo, puedes importar funciones así:

from contador\_objetos.preprocesamiento import convertirBGR\_a\_grises



\### 6. Actualizar el proyecto

Si hay cambios nuevos en el repositorio:

git pull origin main



\## Resultados





El sistema permite:



Detectar objetos en la imagen

Segmentar regiones correctamente

Contar el número total de objetos

Obtener centroides de cada objeto

Detectar esquinas relevantes



Se generan visualizaciones como:



Imagen original

Escala de grises

Bordes (Sobel X, Y y magnitud)

Imagen binaria (umbralización)

Objetos etiquetados

Centroides

Esquinas detectadas



&#x20;Autor

\-ALVARADO ESPARZA JAVIER EMMANUEL

\-FELIX PEREIDA ANDREA

\-ORTIZ MACIAS ATZHYRI GUADALUPE

\-VAZQUEZ MARTINEZ CESAR GIOVANNI

\-Ingeniería Biomédica – Universidad Autónoma de Aguascalientes



&#x20;Licencia

Proyecto de uso académico.

