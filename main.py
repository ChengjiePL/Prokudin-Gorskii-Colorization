import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def load_prokudin_image(filename):
    """Carga y divide la imagen en 3 canales con manejo de residuos"""
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {filename}")
    
    h = img.shape[0] // 3
    total_height = 3 * h
    img = img[:total_height, :]  # Recortar residuo
    return img[:h, :], img[h:2*h, :], img[2*h:3*h, :]

def align_using_ncc(ref, target, search_range=15):
    """Alineación usando Correlación Cruzada Normalizada"""
    best_score = -np.inf
    best_dx, best_dy = 0, 0
    
    # Normalización
    ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)
    target_norm = (target - np.mean(target)) / (np.std(target) + 1e-8)

    for dx in range(-search_range, search_range+1):
        for dy in range(-search_range, search_range+1):
            shifted = np.roll(target_norm, (dx, dy), (0, 1))
            score = np.sum(ref_norm * shifted)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    aligned = np.roll(target, (best_dx, best_dy), (0, 1))
    return aligned, (best_dx, best_dy)

def align_phase_correlation(ref, target):
    """Alineación usando Correlación de Fase con OpenCV"""
    # Aplicar ventana de Hann para reducir artefactos de bordes
    hann = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    
    ref_windowed = np.float32(ref) * hann
    target_windowed = np.float32(target) * hann
    
    # Calcular desplazamiento
    (dx, dy), _ = cv2.phaseCorrelate(ref_windowed, target_windowed)
    return np.roll(target, (int(dy), int(dx)), (0, 1)), (int(dy), int(dx))

def align_with_pyramid(ref, target, method='phase', levels=3):
    """Alineación multiresolución usando pirámide de imágenes"""
    current_ref = ref.copy()
    current_target = target.copy()
    total_dx, total_dy = 0, 0

    for level in range(levels, 0, -1):
        scale = 1 / (2 ** (level-1))
        if level < levels:
            current_ref = cv2.resize(current_ref, None, fx=scale, fy=scale)
            current_target = cv2.resize(current_target, None, fx=scale, fy=scale)

        if method == 'phase':
            aligned, (dx, dy) = align_phase_correlation(current_ref, current_target)
        else:
            aligned, (dx, dy) = align_using_ncc(current_ref, current_target, search_range=5)

        total_dx += dx * (2 ** (level-1))
        total_dy += dy * (2 ** (level-1))

    return np.roll(target, (int(total_dy), int(total_dx)), (int(total_dy), int(total_dx)))

def colorize_prokudin(filename, method='phase', remove_borders=True):
    start_time = time.time()
    
    # 1. Cargar y dividir canales
    B, G, R = load_prokudin_image(filename)
    
    # 2. Alinear usando método seleccionado (G como referencia)
    if method == 'ncc':
        R_aligned, (dx_r, dy_r) = align_using_ncc(G, R)
        B_aligned, (dx_b, dy_b) = align_using_ncc(G, B)
    elif method == 'pyramid':
        R_aligned, (dx_r, dy_r) = align_with_pyramid(G, R)
        B_aligned, (dx_b, dy_b) = align_with_pyramid(G, B)
    else:  # phase correlation por defecto
        R_aligned, (dx_r, dy_r) = align_phase_correlation(G, R)
        B_aligned, (dx_b, dy_b) = align_phase_correlation(G, B)

    # 3. Crear imagen color
    color_img = cv2.merge([B_aligned, G, R_aligned])
    
    # 4. Recorte de bordes dinámico
    if remove_borders:
        def calculate_margin(*displacements):
            return max(max(d for d in displacements), 0)
        
        top = calculate_margin(dy_b, dy_r)
        bottom = calculate_margin(-dy_b, -dy_r)
        left = calculate_margin(dx_b, dx_r)
        right = calculate_margin(-dx_b, -dx_r)
        
        h, w = color_img.shape[:2]
        color_img = color_img[top:h-bottom, left:w-right]

    # 5. Guardar y mostrar resultados
    output_path = filename.replace(".jpg", f"_color_{method}.jpg")
    cv2.imwrite(output_path, color_img)
    
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Resultado ({method})")
    plt.axis('off')
    plt.show()

    print(f"Procesado: {filename}")
    print(f"Desplazamientos B: ({dx_b}, {dy_b}), R: ({dx_r}, {dy_r})")
    print(f"Tiempo: {time.time()-start_time:.2f}s")
    return color_img

def main():
    images = [
        "./prueba/00877v.jpg",
        "./prueba/00974r.jpg",
        "./prueba/00893r.jpg",
        "./prueba/01043v.jpg"
    ]

    # images = [
    #     "./dataset/casa.jpg",
    #     "./dataset/mansion.jpg",
    #     "./dataset/cuadro.jpg",
    #     "./dataset/cruz.jpg",
    #     "./dataset/paisage.jpg",
    # ]

    for img in images:
        try:
            # Probar diferentes métodos
            colorize_prokudin(img, method='phase')
            colorize_prokudin(img, method='pyramid')
        except Exception as e:
            print(f"Error procesando {img}: {str(e)}")

if __name__ == "__main__":
    main()
