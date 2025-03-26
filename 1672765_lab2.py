import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# ====================== MEJORAS ==============================================================
def apply_shift(img, dx, dy):
    """
    Aplica desplazamiento con manejo correcto de bordes usando warpAffine
    """
    rows, cols = img.shape
    M = np.float32([[1, 0, dy], [0, 1, dx]])  # Matriz de transformación
    shifted = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return shifted


def apply_photometric_corrections(img):
    img = img.astype(np.float32) / 255.0

    # Balance de blancos conservador
    avg_B = np.mean(img[:, :, 0])
    avg_G = np.mean(img[:, :, 1])
    avg_R = np.mean(img[:, :, 2])

    # Factor de corrección limitado (evitar cambios bruscos)
    img[:, :, 0] = np.clip(img[:, :, 0] * (0.5 + avg_G / (2 * avg_B)), 0, 1)
    img[:, :, 2] = np.clip(img[:, :, 2] * (0.5 + avg_G / (2 * avg_R)), 0, 1)

    # Ecualización adaptativa (mejor que global)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img[:, :, i] = clahe.apply((img[:, :, i] * 255).astype(np.uint8)) / 255.0

    return (img * 255).astype(np.uint8)


def remove_artifacts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detección más precisa de defectos
    blurred = cv2.medianBlur(gray, 5)
    _, mask = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)  # Reducir umbral

    # Operaciones morfológicas más conservadoras
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)  # Erosionar antes de dilatar
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpainting con método Navier-Stokes (menos borroso)
    result = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)

    return result


# ====================== CARGA DE LA IMAGEN PROKUDIN CON RECORTE INICIAL ======================
def load_prokudin_image(filename):
    """
    Carga la placa Prokudin-Gorskii en escala de grises con recorte inicial
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {filename}")

    # Recorte inicial de bordes (1% arriba, 2% abajo, 5% lados)
    h, w = img.shape
    img = img[int(h * 0.01) : int(h - h * 0.02), int(w * 0.05) : int(w - w * 0.05)]

    # Dividir en canales
    h = (
        img.shape[0] // 3
    )  # cálculo dinámico de la altura de los canales, para manejar cualquier tipo de tamaño
    B = img[0:h, :]
    G = img[h : 2 * h, :]
    R = img[2 * h : 3 * h, :]

    return B, G, R


# ====================== NCC MEJORADA ======================
def ncc(a, b):
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    return np.sum(a_mean * b_mean) / (np.linalg.norm(a_mean) * np.linalg.norm(b_mean))


def align_using_ncc(ref, target, search_range=20):
    """
    Versión mejorada de NCC basada en el segundo código
    """
    max_score = -np.inf
    best_shift = (0, 0)

    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            shifted = np.roll(target, (dx, dy), axis=(0, 1))
            current_ncc = ncc(ref, shifted)
            if current_ncc > max_score:
                max_score = current_ncc
                best_shift = (dx, dy)

    aligned = np.roll(target, best_shift, axis=(0, 1))
    return aligned, best_shift


def align_corr_space(ref, target, search_range=15):
    """
    Alineamiento de imágenes usando correlación basada en convolución en el espacio.
    Se desplaza la imagen en un rango dado y se calcula la correlación con la referencia.
    """
    best_score = -np.inf
    best_dx, best_dy = 0, 0

    # Convertir a escala de grises si es necesario

    if len(ref.shape) == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    # Suavizado ligero para reducir ruido

    ref_blur = cv2.GaussianBlur(ref, (5, 5), 0)
    target_blur = cv2.GaussianBlur(target, (5, 5), 0)

    # Búsqueda de la mejor alineación

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            shifted = np.roll(target_blur, (dy, dx), axis=(0, 1))  # Desplazar imagen
            score = np.sum(ref_blur * shifted)  # Producto interno (correlación)

            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
    # Aplicar desplazamiento final

    aligned = np.roll(target, (best_dy, best_dx), axis=(0, 1))

    return aligned, (best_dx, best_dy)


# ====================== 2) Correlación en Fourier (producto directo) ======================
def align_corr_fourier(ref, target):
    """
    Correlación en el dominio de Fourier:
    C = IFFT( FFT(ref) * conj(FFT(target)) )
    y buscar el máximo.
    """
    # Convertir a float64
    ref_f = np.float64(ref)
    tar_f = np.float64(target)

    # FFT2 de ambas
    Fref = np.fft.fft2(ref_f)
    Ftar = np.fft.fft2(tar_f)

    # Cross-correlation en frecuencia
    cross_corr = np.fft.ifft2(Fref * np.conj(Ftar))

    # Localizar el máximo
    # cross_corr es complejo, tomamos magnitud real
    cross_corr_mag = np.abs(cross_corr)

    # Índice del máximo
    max_idx = np.unravel_index(np.argmax(cross_corr_mag), cross_corr_mag.shape)
    peak_y, peak_x = max_idx

    # Ajustar desplazamiento (peak) a rango negativo/positivo
    h, w = ref.shape
    # Convertimos el pico a dx,dy
    # Si peak_x > w//2, el desplazamiento es peak_x - w
    # Si peak_y > h//2, el desplazamiento es peak_y - h
    dy = int(peak_y if peak_y <= h // 2 else peak_y - h)
    dx = int(peak_x if peak_x <= w // 2 else peak_x - w)

    aligned = np.roll(target, (dy, dx), axis=(0, 1))
    return aligned, (dy, dx)


# ====================== 3) Correlación de fase (basada en Fourier) ======================
def align_phase_correlation(ref, target):
    ref_f = np.float32(ref)
    tar_f = np.float32(target)

    # Máscara adaptativa basada en energía espectral
    h, w = ref.shape
    grad_x = cv2.Sobel(ref, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(ref, cv2.CV_32F, 0, 1)
    energy = cv2.magnitude(grad_x, grad_y)
    _, mask = cv2.threshold(energy, 0.5 * np.max(energy), 1.0, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (25, 25), 0).astype(np.float32)

    (dx, dy), _ = cv2.phaseCorrelate(ref_f, tar_f, mask)

    # Ajuste subpixel para mayor precisión
    dx_fine = dx - int(dx)
    dy_fine = dy - int(dy)
    aligned = apply_shift(target, int(dx), int(dy))
    aligned = apply_shift(aligned, dx_fine, dy_fine)  # Desplazamiento fraccional

    return aligned, (dx, dy)


# ====================== FUNCIÓN PRINCIPAL MODIFICADA CON RECORTE FINAL ======================
def colorize_prokudin(filename, method="ncc", search_range=20):
    """
    Función principal con recorte final de bordes
    """
    start_time = time.time()

    B, G, R = load_prokudin_image(filename)

    # Selección de método
    if method == "corr_space":
        align_func = lambda ref, tgt: align_corr_space(ref, tgt, search_range)
    elif method == "fourier":
        align_func = align_corr_fourier
    elif method == "phase":
        align_func = align_phase_correlation
    elif method == "ncc":
        align_func = lambda ref, tgt: align_using_ncc(ref, tgt, search_range)
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Alinear canales
    R_aligned, (dx_r, dy_r) = align_func(G, R)
    B_aligned, (dx_b, dy_b) = align_func(G, B)

    # Crear imagen color
    # color_img = cv2.merge([B_aligned, G, R_aligned])
    color_img = cv2.merge([apply_shift(B, dx_b, dy_b), G, apply_shift(R, dx_r, dy_r)])

    # Recorte final de bordes (10%)
    border = int(min(color_img.shape[:2]) * 0.10)
    color_img = color_img[
        int(border) : int(color_img.shape[0] - border),
        int(border) : int(color_img.shape[1] - border),
    ]
    # Aplicar mejoras
    color_img = apply_photometric_corrections(color_img)
    color_img = remove_artifacts(color_img)

    # Guardar y mostrar resultados
    output_path = filename.replace(".jpg", f"_{method}.jpg")
    cv2.imwrite(output_path, color_img)

    elapsed = time.time() - start_time
    print(f"Procesado: {filename}")
    print(f"Método: {method} => Desplazamientos B=({dx_b},{dy_b}), R=({dx_r},{dy_r})")
    print(f"Tiempo: {elapsed:.2f}s | Guardado en: {output_path}")

    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Resultado {method}")
    plt.axis("off")
    plt.show()


def main():
    images = [
        "./prueba/01044r.jpg",
        # "./prueba/00877v.jpg",
        # "./prueba/00974r.jpg",
        # "./prueba/00893r.jpg",
        # "./prueba/01043v.jpg",
        # "./dataset/casa.jpg",
        # "./dataset/mansion.jpg",
        # "./dataset/paisage.jpg",
        # "./dataset/cuadro.jpg",
        # "./dataset/cruz.jpg",
    ]

    methods = ["corr_space", "fourier", "phase", "ncc"]

    for img_path in images:
        for m in methods:
            try:
                colorize_prokudin(img_path, method=m, search_range=20)
            except Exception as e:
                print(f"Error procesando {img_path} con {m}: {str(e)}")


if __name__ == "__main__":
    main()
