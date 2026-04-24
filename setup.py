# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:36:54 2026

@author: emman
"""

from setuptools import setup, find_packages

setup(
    name="contador_objetos",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "opencv-python"
    ],
    author="Javier Emmanuel Alvarado",
    description="Librería para segmentación y conteo de objetos en imágenes",
)