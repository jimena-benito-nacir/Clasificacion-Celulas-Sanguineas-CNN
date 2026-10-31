# Clasificación de células sanguíneas con redes neuronales convolucionales

Trabajo práctico de Deep Learning — **Diplomatura en Ciencia de Datos y Análisis Avanzado, Módulo 8**
Grupo 35: **M. Jimena Benito Nacir** · Septiembre 2026

## Objetivo

Desarrollar una CNN propia y compararla con un modelo preentrenado (VGG16) adaptado mediante transferencia de aprendizaje, para clasificar imágenes microscópicas de células sanguíneas en ocho categorías.

## Dataset

Con autorización del docente se utilizó **[BloodMNIST](https://medmnist.com/)** (colección MedMNIST) en lugar de Intel Image Classification: 17.092 imágenes en color de 64 × 64 píxeles, en ocho categorías: basófilos, eosinófilos, eritroblastos, granulocitos inmaduros, linfocitos, monocitos, neutrófilos y plaquetas. Se conservó la división original (70 % entrenamiento, 10 % validación, 20 % prueba).

**Depuración:** la comparación píxel a píxel detectó 17 imágenes repetidas (8 dentro de entrenamiento y 9 compartidas entre conjuntos) y una imagen con etiquetas contradictorias (eosinófilo en entrenamiento, neutrófilo en prueba). Se excluyeron 18 imágenes para evitar fuga de información.

## Métrica

**F1 macro** como métrica principal (las clases están desbalanceadas: neutrófilos 19,5 % vs. linfocitos 7,1 %). Como métricas auxiliares: precisión y recall por clase, matriz de confusión y análisis visual de errores.

## Modelos

| Modelo | Descripción |
|---|---|
| **CNN propia** | 3 bloques Conv2D + MaxPooling (32 → 64 → 128 filtros), Dense 128 + Dropout 0,5, salida softmax de 8 clases |
| **VGG16 congelada** | Base preentrenada en ImageNet congelada + GlobalAveragePooling, Dense 128, Dropout y salida de 8 clases |
| **VGG16 con ajuste fino** | Igual a la anterior, con el bloque convolucional 5 descongelado y tasa de aprendizaje reducida |

Entrenamiento con Adam, `sparse_categorical_crossentropy`, lotes de 32, `EarlyStopping` y `ModelCheckpoint` sobre la pérdida de validación.

## Resultados (conjunto de prueba, 3.420 imágenes)

| Modelo | F1 macro | Pérdida | Aciertos | Errores | Parámetros | Tiempo de entrenamiento* |
|---|---:|---:|---:|---:|---:|---:|
| **CNN propia** | **0,9562** | **0,1324** | 3.287 | 133 | 1.142.984 | 78 s |
| VGG16 con ajuste fino | 0,9220 | 0,2491 | 3.173 | 247 | 14.781.384 | 396 s |
| VGG16 congelada | 0,8650 | 0,3728 | 2.992 | 428 | 14.781.384 | 245 s |

\*GPU T4 en Google Colab. El tiempo del ajuste fino incluye la etapa previa con la base congelada.

La CNN propia obtuvo el mejor F1 en las ocho categorías con el menor costo computacional. Las categorías más difíciles para los tres modelos fueron monocitos y granulocitos inmaduros; la mejor reconocida, las plaquetas.

## Conclusión

Se selecciona la **CNN propia** como modelo final. La transferencia de aprendizaje con VGG16 no superó a una red diseñada para el problema, aun con ajuste fino: con entradas de 64 × 64, VGG16 reduce las imágenes a mapas de 2 × 2, y las características aprendidas en ImageNet solo son parcialmente transferibles a imágenes microscópicas de células teñidas.

## Estructura del repositorio

```
├── notebooks/
│   └── TP_Modulo8_CNN_BenitoNacir_MariaJimena.ipynb   # Notebook completo con código, salidas e interpretaciones
├── informe/
│   └── TP_DL_DiplomaturaCienciaDeDatos_Modulo8_BenitoNacir_MariaJimena.pdf   # Informe final
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

El notebook fue desarrollado en **Google Colab** con GPU T4.

1. Abrir `notebooks/TP_Modulo8_CNN_BenitoNacir_MariaJimena.ipynb` en Colab.
2. Seleccionar un entorno con GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución*).
3. Ejecutar las celdas en orden. El notebook monta Google Drive y usa la carpeta `/content/drive/MyDrive/TP8_BloodMNIST` para guardar el dataset y los modelos entrenados.

Para ejecutarlo localmente: `pip install -r requirements.txt` y adaptar las celdas de montaje de Google Drive a una carpeta local.

## Bibliografía

- Acevedo, A. et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. *Data in Brief, 30*, 105474. https://doi.org/10.1016/j.dib.2020.105474
- Deng, J. et al. (2009). ImageNet: A large-scale hierarchical image database. *IEEE CVPR*, 248–255.
- James, G., Witten, D., Hastie, T. y Tibshirani, R. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
- Simonyan, K. y Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*. https://arxiv.org/abs/1409.1556
- Yang, J. et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data, 10*, 41.
