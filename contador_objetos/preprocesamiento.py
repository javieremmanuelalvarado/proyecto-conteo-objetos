# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:28:49 2026

@author: emman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def leer_imagen(ruta_imagen):
    """
    Lee una imagen desde disco usando OpenCV.
    OpenCV la carga en formato BGR.
    """
    imagen = cv2.imread(ruta_imagen)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {ruta_imagen}")

    return imagen


def bgr_a_rgb(imagen_bgr):
    """
    Convierte una imagen de BGR a RGB manualmente.
    """
    imagen_rgb = imagen_bgr[:, :, ::-1]
    return imagen_rgb


def convertirBGR_a_grises(imagen_bgr):
    """
    Convierte una imagen BGR a escala de grises manualmente.
    Fórmula usada:
        gris = 0.2989*R + 0.587*G + 0.114*B
    """
    b = imagen_bgr[:, :, 0].astype(np.float64)
    g = imagen_bgr[:, :, 1].astype(np.float64)
    r = imagen_bgr[:, :, 2].astype(np.float64)

    gris = 0.114 * b + 0.587 * g + 0.2989 * r

    gris = np.clip(gris, 0, 255)
    gris = gris.astype(np.uint8)

    return gris

def convertirRGB_a_grises(imagen_rgb):
    """
    Convierte una imagen RGB a escala de grises manualmente.
    Fórmula usada:
        gris = 0.2989*R + 0.587*G + 0.114*B
    """
    r = imagen_rgb[:, :, 0].astype(np.float64)
    g = imagen_rgb[:, :, 1].astype(np.float64)
    b = imagen_rgb[:, :, 2].astype(np.float64)

    gris = 0.114 * b + 0.587 * g + 0.2989 * r

    gris = np.clip(gris, 0, 255)
    gris = gris.astype(np.uint8)

    return gris

def aplicar_convolucion(imagen, kernel, aplicar_clip=True):
    """
    Aplica una convolución manual a una imagen en escala de grises.
    
    Parámetros:
    - imagen: matriz 2D
    - kernel: matriz del filtro
    - aplicar_clip: si True, limita la salida entre 0 y 255
    """

    alto, ancho = imagen.shape
    k_alto, k_ancho = kernel.shape

    pad_h = k_alto // 2
    pad_w = k_ancho // 2

    imagen_padded = np.pad(imagen,
                           ((pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    salida = np.zeros((alto, ancho), dtype=np.float64)

    for i in range(alto):
        for j in range(ancho):
            region = imagen_padded[i:i + k_alto, j:j + k_ancho]
            valor = np.sum(region * kernel)
            salida[i, j] = valor

    if aplicar_clip:
        salida = np.clip(salida, 0, 255)
        return salida.astype(np.uint8)

    return salida

def filtro_promedio():
    kernel = np.ones((3,3))
    return kernel / np.sum(kernel)


def kernel_sobel_x():
    """
    Devuelve el kernel de Sobel en dirección X.
    """
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def kernel_sobel_y():
    """
    Devuelve el kernel de Sobel en dirección Y.
    """
    return np.array([[-1, -2, -1],[ 0,  0,  0], [ 1,  2,  1]], dtype=np.float64)


def calcular_magnitud_gradiente(gradiente_x, gradiente_y):
    """
    Calcula la magnitud del gradiente combinando Gx y Gy.
    """
    magnitud = np.sqrt(gradiente_x.astype(np.float64)**2 +
                       gradiente_y.astype(np.float64)**2)

    magnitud = np.clip(magnitud, 0, 255)
    return magnitud.astype(np.uint8)

def transformacion_negativa(imagen):
    """
    Aplica la transformación negativa a una imagen en escala de grises.
    
    Parámetro:
    - imagen: imagen en escala de grises (uint8)
    
    Devuelve:
    - imagen negativa
    """
    negativo = 255 - imagen
    return negativo


def calcular_matriz_harris(imagen_gris, k=0.04):
    """
    Calcula la respuesta de Harris para una imagen en escala de grises.
    
    Parámetros:
    - imagen_gris: imagen en escala de grises
    - k: constante de Harris
    
    Devuelve:
    - matriz R con la respuesta de Harris
    """
    sobel_x = kernel_sobel_x()
    sobel_y = kernel_sobel_y()

    ix = aplicar_convolucion(imagen_gris, sobel_x, aplicar_clip=False)
    iy = aplicar_convolucion(imagen_gris, sobel_y, aplicar_clip=False)

    ix2 = ix ** 2
    iy2 = iy ** 2
    ixy = ix * iy

    kernel = filtro_promedio()

    sx2 = aplicar_convolucion(ix2, kernel, aplicar_clip=False)
    sy2 = aplicar_convolucion(iy2, kernel, aplicar_clip=False)
    sxy = aplicar_convolucion(ixy, kernel, aplicar_clip=False)

    det_m = (sx2 * sy2) - (sxy ** 2)
    trace_m = sx2 + sy2

    r = det_m - k * (trace_m ** 2)

    return r


def detectar_esquinas_harris(respuesta_harris, umbral_relativo=0.01):
    """
    Detecta esquinas a partir de la respuesta de Harris.
    
    Parámetros:
    - respuesta_harris: matriz R
    - umbral_relativo: porcentaje respecto al valor máximo
    
    Devuelve:
    - lista de coordenadas (fila, columna) donde hay esquinas
    """
    esquinas = []

    r_max = np.max(respuesta_harris)
    umbral = umbral_relativo * r_max

    alto, ancho = respuesta_harris.shape

    for i in range(1, alto - 1):
        for j in range(1, ancho - 1):
            valor = respuesta_harris[i, j]

            if valor > umbral:
                ventana = respuesta_harris[i-1:i+2, j-1:j+2]

                if valor == np.max(ventana):
                    esquinas.append((i, j))

    return esquinas