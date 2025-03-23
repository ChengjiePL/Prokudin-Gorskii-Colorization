
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ====================== CARGA DE LA IMAGEN PROKUDIN (SI PROCEDE) ======================
def load_prokudin_image(filename):
    """
    Carga la placa Prokudin-Gorskii en escala de grises y la divide en B, G, R.
    Si no es Prokudin, puedes omitir esta función.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {filename}")
    # Suponemos 3 bandas verticales
    h = img.shape[0] // 3
    B = img[0:h, :]
    G = img[h:2*h, :]
    R = img[2*h:3*h, :]
    return B, G, R

# ====================== 1) Correlación en el espacio (convolución) ======================
def align_corr_space(ref, target, search_range=15):
    """
    Correlación basada en convolución en el espacio:
    Recorre un rango de desplazamientos y calcula la suma de productos (score).
    Sin normalizar.
    """
    best_score = -np.inf
    best_dx, best_dy = 0, 0
    
    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            # Desplazamos 'target'
            shifted = np.roll(target, (dx, dy), axis=(0,1))
            # Suma de productos directos (convolución en el espacio)
            score = np.sum(ref * shifted)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    aligned = np.roll(target, (best_dx, best_dy), axis=(0,1))
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
    dy = int(peak_y if peak_y <= h//2 else peak_y - h)
    dx = int(peak_x if peak_x <= w//2 else peak_x - w)

    aligned = np.roll(target, (dy, dx), axis=(0,1))
    return aligned, (dy, dx)

# ====================== 3) Correlación de fase (basada en Fourier) ======================
def align_phase_correlation(ref, target):
    """
    Usa la función de OpenCV phaseCorrelate() o una implementación manual.
    Aquí implementamos con OpenCV.
    """
    ref_f = np.float32(ref)
    tar_f = np.float32(target)
    
    # PhaseCorrelate
    (dx, dy), _ = cv2.phaseCorrelate(ref_f, tar_f)
    # Note: phaseCorrelate retorna (dx,dy) invertidos
    # Ajustamos para que sea (dy,dx) o lo que corresponda
    # Realmente la doc dice (shift.x, shift.y) => (dx, dy)
    # Suele significar que hay que usar roll(target, (int(dy), int(dx)))
    aligned = np.roll(target, (int(round(dy)), int(round(dx))), axis=(0,1))
    return aligned, (int(round(dy)), int(round(dx)))

# ====================== 4) Correlación Normalizada (NCC) ======================
def align_using_ncc(ref, target, search_range=15):
    """
    Alineación usando Correlación Cruzada Normalizada (NCC).
    """
    best_score = -np.inf
    best_dx, best_dy = 0, 0
    
    # Normalizar ref y target
    ref_mean, ref_std = np.mean(ref), np.std(ref) + 1e-8
    target_mean, target_std = np.mean(target), np.std(target) + 1e-8
    
    ref_norm = (ref - ref_mean) / ref_std
    
    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            shifted = np.roll(target, (dx, dy), axis=(0,1))
            shifted_norm = (shifted - target_mean) / target_std
            score = np.sum(ref_norm * shifted_norm)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    aligned = np.roll(target, (best_dx, best_dy), axis=(0,1))
    return aligned, (best_dx, best_dy)

# ====================== EJEMPLO PRINCIPAL ======================
def colorize_prokudin(filename, method='corr_space', search_range=15):
    """
    Carga la placa Prokudin-Gorskii, alinea canales (G como referencia) 
    usando uno de los 4 métodos:
      - corr_space  (convolución en el espacio)
      - fourier     (producto en el espacio de Fourier)
      - phase       (correlación de fase)
      - ncc         (correlación normalizada)
    y genera la imagen color resultante.
    """
    start_time = time.time()
    
    B, G, R = load_prokudin_image(filename)
    
    # Elegimos la función de alineación
    if method == 'corr_space':
        align_func = lambda ref, tgt: align_corr_space(ref, tgt, search_range)
    elif method == 'fourier':
        align_func = align_corr_fourier
    elif method == 'phase':
        align_func = align_phase_correlation
    elif method == 'ncc':
        align_func = lambda ref, tgt: align_using_ncc(ref, tgt, search_range)
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Alinear R y B respecto a G
    R_aligned, (dx_r, dy_r) = align_func(G, R)
    B_aligned, (dx_b, dy_b) = align_func(G, B)
    
    # Fusionar canales
    color_img = cv2.merge([B_aligned, G, R_aligned])
    
    # Guardar
    output_path = filename.replace(".jpg", f"_{method}.jpg")
    cv2.imwrite(output_path, color_img)
    
    elapsed = time.time() - start_time
    print(f"Procesado: {filename}")
    print(f"Método: {method} => Desplazamiento B=({dx_b},{dy_b}), R=({dx_r},{dy_r})")
    print(f"Tiempo: {elapsed:.2f} s => Guardado en {output_path}")
    
    # Visualizar
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Resultado {method}")
    plt.axis('off')
    plt.show()

def main():
    images = [
        "./prueba/00877v.jpg",
        "./prueba/00974r.jpg",
        "./prueba/00893r.jpg",
        "./prueba/01043v.jpg",
        "./dataset/casa.jpg",
        "./dataset/mansion.jpg",
        "./dataset/paisage.jpg",
        "./dataset/cuadro.jpg",
        "./dataset/cruz.jpg",
    ]
    
    methods = ['corr_space', 'fourier', 'phase', 'ncc']
    
    for img_path in images:
        for m in methods:
            try:
                colorize_prokudin(img_path, method=m, search_range=15)
            except Exception as e:
                print(f"Error procesando {img_path} con {m}: {str(e)}")

if __name__ == "__main__":
    main()

