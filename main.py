import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def load_prokudin_image(filename):
    """
    Carga la placa de Prokudin-Gorskii (en escala de grises)
    y devuelve 3 subimágenes: B, G, R.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    h = img.shape[0] // 3  # se asume que la imagen es 3x la altura de un canal

    # Extraer subimágenes
    B = img[0:h, :]
    G = img[h : 2 * h, :]
    R = img[2 * h : 3 * h, :]

    return B, G, R


def align_using_correlation(ref, target, search_range=15):
    """
    Alinea 'target' respecto a 'ref' buscando el desplazamiento
    que maximiza la correlación. search_range define el rango de pixeles a desplazar.
    Devuelve la imagen alineada y el desplazamiento (dx, dy).
    """
    best_score = -1e9
    best_dx, best_dy = 0, 0

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            # Desplazar target
            shifted = np.roll(target, shift=dx, axis=0)
            shifted = np.roll(shifted, shift=dy, axis=1)

            # Calcular correlación (ej. suma de productos)
            score = np.sum(ref * shifted)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    # Crear la imagen final con el mejor desplazamiento
    aligned = np.roll(target, shift=best_dx, axis=0)
    aligned = np.roll(aligned, shift=best_dy, axis=1)

    return aligned, (best_dx, best_dy)


def align_phase_correlation(ref, target):
    # Convertir a float32
    ref_f = np.float32(ref)
    tgt_f = np.float32(target)
    # OpenCV tiene la función cv2.phaseCorrelate en C++,
    # en Python se puede usar np.fft.fft2 manualmente.
    # ...
    # Devuelve la imagen alineada y el desplazamiento
    pass


def colorize_prokudin(filename, method="correlation", search_range=15):
    """
    Crea una imagen a color a partir de la placa Prokudin-Gorskii
    y la guarda como 'filename_color.jpg'.
    method puede ser 'correlation' o 'phase'.
    """
    start_time = time.time()

    # 1) Cargar y recortar
    B, G, R = load_prokudin_image(filename)

    # 2) Alinear. Elegimos R como referencia
    if method == "correlation":
        G_aligned, (dx_g, dy_g) = align_using_correlation(R, G, search_range)
        B_aligned, (dx_b, dy_b) = align_using_correlation(R, B, search_range)
    else:
        # Alinear con phase correlation (opcional)
        G_aligned, (dx_g, dy_g) = align_phase_correlation(R, G)
        B_aligned, (dx_b, dy_b) = align_phase_correlation(R, B)

    # 3) Fusionar canales (OpenCV usa BGR)
    color_img = cv2.merge([B_aligned, G_aligned, R])

    # 4) Guardar resultado
    out_name = filename.replace(".jpg", "_color.jpg")
    cv2.imwrite(out_name, color_img)

    elapsed = time.time() - start_time
    print(f"Procesado: {filename}")
    print(f"Desplazamientos G: ({dx_g}, {dy_g}), B: ({dx_b}, {dy_b})")
    print(f"Tiempo total: {elapsed:.2f} s")
    print(f"Guardado en: {out_name}")

    # Mostrar
    plt.imshow(color_img, cmap="gray")  # BGR => si quieres color real, usa cvtColor
    plt.title("Resultado Color")
    plt.axis("off")
    plt.show()

    return color_img


def remove_borders(img, top=10, bottom=10, left=10, right=10):
    h, w = img.shape[:2]
    plt.imshow(img[top : h - bottom, left : w - right], cmap="gray")
    plt.title("Imagen sin bordes")
    plt.axis("off")
    plt.show()

    return img[top : h - bottom, left : w - right]


def main():
    # Lista de imágenes de Prokudin-Gorskii
    images = [
        "./dataset/casa.jpg",
        "./dataset/mansion.jpg",
        "./dataset/cuadro.jpg",
        "./dataset/cruz.jpg",
        "./dataset/paisage.jpg",
    ]

    for img_file in images:
        color_img = colorize_prokudin(img_file, method="correlation", search_range=15)
        remove_borders(color_img, top=10, bottom=10, left=10, right=10)


if __name__ == "__main__":
    main()
