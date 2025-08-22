import numpy as np
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image

#image = Image.open()
# Cargar la imagen (reemplaza 'ruta/a/tu/imagen.jpg' con la ruta real de tu archivo)
# El 1 indica que la imagen se cargará en color
ruta = r"C:\Users\GMADRO04\Documents\SOLER\QUPA\Testeo_QUPA\camara_QUPA\img_1280_curadas\img_20250726_064627.jpg"
imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Mostrar la imagen en una ventana llamada "Mi Imagen"
    cv2.imshow('Mi Imagen', imagen)

    # Esperar indefinidamente hasta que el usuario presione una tecla
    cv2.waitKey(0)

    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()
# Crea una máscara circular centrada en la imagen

# Define las dimensiones de la imagen y el centro del círculo
ancho, alto = 1280, 1080 # pixeles
# centro_x, centro_y = ancho // 2, alto // 2
centro_x, centro_y = 574.250000000000,alto // 2
radio = 243.468298757764

# Crea un array de coordenadas
y, x = np.ogrid[:alto, :ancho]

# Calcula la distancia de cada punto al centro
distancia_al_centro = np.sqrt((x - centro_x)**2 + (y - centro_y)**2)

# Crea la máscara booleana: True donde la distancia es menor que el radio
mascara_circulo = distancia_al_centro <= radio

# Puedes convertir la máscara booleana a una máscara binaria (0s y 1s) si lo necesitas
mascara_binaria = mascara_circulo.astype(np.uint8) * 255 # 255 para blanco

masked = cv2.bitwise_and(imagen, imagen, mask=mascara_binaria)
cv2.imshow('Mascara Circular', mascara_binaria)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Imagen con Máscara Circular', masked)
cv2.waitKey(0)  