# 🧠 Conteo de Objetos en Imágenes  
### Procesamiento de Imágenes Biomédicas

Proyecto desarrollado en Python para la segmentación, análisis y conteo de objetos en imágenes utilizando técnicas fundamentales implementadas manualmente.



## 📌 Descripción

Este proyecto implementa un pipeline completo de procesamiento de imágenes que incluye:

- Conversión de RGB a escala de grises  
- Filtrado espacial (convolución manual)  
- Transformaciones de intensidad (negativo)  
- Detección de bordes (Sobel)  
- Umbralización  
- Segmentación por componentes conectados  
- Eliminación de objetos pequeños (ruido)  
- Cálculo de propiedades de objetos (área, centroides)  
- Detección de esquinas (Harris)  
- Conteo final de objetos  

El enfoque principal es **no depender de funciones automáticas**, sino implementar los algoritmos desde cero para entender su funcionamiento.



## 🛠️ Tecnologías utilizadas

- Python  
- NumPy  
- Matplotlib  
- OpenCV (uso mínimo)



## 📂 Estructura del proyecto
proyecto-conteo-objetos/
│
├── contador_objetos/
│ ├── preprocesamiento.py
│ ├── segmentacion.py
│ ├── caracteristicas.py
│ ├── pipeline.py
│ └── init.py
│
├── ejemplos/
│ └── demo.py
│
├── imagenes_prueba/
│ └── (imágenes de prueba)
│
├── requirements.txt
├── setup.py
└── README.md

2. Instalar dependencias
pip install -r requirements.txt

3. Ejecutar el ejemplo
python ejemplos/demo.py

🧪 Resultados

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

👨‍💻 Autor
ALVARADO ESPARZA JAVIER EMMANUEL
FELIX PEREIDA ANDREA
ORTIZ MACIAS ATZHYRI GUADALUPE
VAZQUEZ MARTINEZ CESAR GIOVANNI
Ingeniería Biomédica – Universidad Autónoma de Aguascalientes

📄 Licencia
Proyecto de uso académico.
