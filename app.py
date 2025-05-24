import os
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import cv2
import numpy as np
from scipy.signal import wiener

app = Flask(__name__)

# Folder struktur noise
NOISE_UPLOAD_FOLDER = 'noise/upload'
NOISE_RESULT_FOLDER = 'noise/result'
# Folder struktur restore
RESTORE_UPLOAD_FOLDER = 'restore/upload'
RESTORE_RESULT_FOLDER = 'restore/result'

for folder in [NOISE_UPLOAD_FOLDER, NOISE_RESULT_FOLDER, RESTORE_UPLOAD_FOLDER, RESTORE_RESULT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Fungsi tambah noise
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_gaussian_noise(image, mean=0, sigma=20):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_periodic_noise(F, a=3, bx=3, by=5):
    m, n = F.shape
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    derau = a * np.sin(X / bx + Y / by) + 1
    G = np.uint8(np.clip(F + derau, 0, 255))
    return G

# Fungsi filter notch Butterworth
def filnotch(a, b, d0, x, y, n=1):
    u = np.arange(0, a)
    v = np.arange(0, b)
    u[u > b / 2] -= b
    v[v > a / 2] -= a
    V, U = np.meshgrid(v, u)
    D = np.sqrt(V**2 + U**2)
    Hlpf = 1 / (1 + (D / d0)**(2 * n))
    Hhpf = 1 - Hlpf
    H = np.roll(Hhpf, shift=(y - 1, x - 1), axis=(0, 1))
    return H

def remove_periodic_noise(img, notch_centers, d0=10, order=1):
    a, b = img.shape
    r = np.ceil(np.log2(2 * max(a, b)))
    p = int(2 ** r)
    q = p
    F = np.fft.fft2(img, (p, q))
    H = np.ones((p, q))
    for (x, y) in notch_centers:
        H *= filnotch(p, q, d0, x, y, order)
    F_filtered = F * H
    img_back = np.real(np.fft.ifft2(F_filtered))
    img_restored = img_back[:a, :b]
    img_restored = np.clip(img_restored, 0, 255).astype(np.uint8)
    return img_restored

# Fungsi filter lain dan adaptive median
def adaptive_median_filter(img, max_size=7):
    import copy
    filtered = copy.deepcopy(img)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            k = 3
            while k <= max_size:
                x_min = max(i - k//2, 0)
                x_max = min(i + k//2, rows -1)
                y_min = max(j - k//2, 0)
                y_max = min(j + k//2, cols -1)
                window = img[x_min:x_max+1, y_min:y_max+1]
                Z_med = np.median(window)
                Z_min = np.min(window)
                Z_max = np.max(window)
                A1 = Z_med - Z_min
                A2 = Z_med - Z_max
                if A1 > 0 and A2 < 0:
                    B1 = img[i,j] - Z_min
                    B2 = img[i,j] - Z_max
                    if B1 > 0 and B2 < 0:
                        filtered[i,j] = img[i,j]
                    else:
                        filtered[i,j] = Z_med
                    break
                else:
                    k += 2
                    if k > max_size:
                        filtered[i,j] = Z_med
                        break
    return filtered

def apply_filter(img, noise_type, filter_type):
    if noise_type == 'salt_pepper':
        if filter_type == 'median':
            return cv2.medianBlur(img, 3)
        elif filter_type == 'adaptive_median':
            return adaptive_median_filter(img)
    elif noise_type == 'gaussian':
        if filter_type == 'gaussian':
            return cv2.GaussianBlur(img, (5,5), 0)
        elif filter_type == 'wiener':
            from scipy.signal import wiener
            return wiener(img, (5,5)).astype(np.uint8)
    elif noise_type == 'periodic':
        if filter_type == 'notch':
            a, b = img.shape
            r = np.ceil(np.log2(2 * max(a, b)))
            p = int(2 ** r)
            q = p

            H1 = filnotch(p, q, 10, 1, 165)
            H2 = filnotch(p, q, 10, 165, 1)
            H3 = filnotch(p, q, 10, 1, 860)
            H4 = filnotch(p, q, 10, 862, 1)

            F = np.fft.fft2(img, (p, q))

            F_frek = F * H1 * H2 * H3 * H4

            F_hasil = np.real(np.fft.ifft2(F_frek))

            F_hasil = F_hasil[:a, :b]

            return F_hasil
        elif filter_type == 'fourier':
            # Bisa pakai low pass filter atau filter lain
            # Untuk sederhana gunakan same remove_periodic_noise saja atau implementasi lain
            notch_centers = [(1, 165), (165, 1), (1, 860), (862, 1)]
            return remove_periodic_noise(img, notch_centers, d0=20, order=2)
    return img

def calculate_mse(img1, img2):
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return err

def save_image(img, filename, folder):
    path = os.path.join(folder, filename)
    cv2.imwrite(path, img)
    return path

@app.route('/')
def index():
    return render_template('index.html', active='index')

@app.route('/restore')
def restore_page():
    return render_template('restore.html', active='restore')

@app.route('/noisy')
def noisy_page():
    return render_template('noisy.html', active='noisy')

@app.route('/add_noise', methods=['POST'])
def add_noise():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    noise_type = request.form.get('noise_type')

    filepath = os.path.join(NOISE_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500

    if noise_type == 'salt_pepper':
        noisy_img = add_salt_and_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
    elif noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(img, mean=0, sigma=20)
    elif noise_type == 'periodic':
        noisy_img = add_periodic_noise(img, 10, 1, 10000000)
        noisy_img = add_periodic_noise(noisy_img, 10, 100000, -1)
    else:
        noisy_img = img

    noisy_filename = f"{os.path.splitext(file.filename)[0]}_noisy.png"
    save_image(noisy_img, noisy_filename, NOISE_RESULT_FOLDER)

    height, width = img.shape

    return jsonify({
        'original': '/noise/upload/' + file.filename,
        'noisy': '/noise/result/' + noisy_filename,
        'width': width,
        'height': height
    })

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    noise_type = request.form.get('noise_type')
    filter_type = request.form.get('filter_type')

    filepath = os.path.join(RESTORE_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500

    filtered_img = apply_filter(img, noise_type, filter_type)
    saved_filename = f"{os.path.splitext(file.filename)[0]}_filtered.png"
    save_image(filtered_img, saved_filename, RESTORE_RESULT_FOLDER)

    mse = calculate_mse(img, filtered_img)
    height, width = img.shape

    return jsonify({
        'original': '/restore/upload/' + file.filename,
        'filtered': '/restore/result/' + saved_filename,
        'width': width,
        'height': height,
        'mse': mse
    })

@app.route('/noise/upload/<filename>')
def noise_uploaded_file(filename):
    return send_from_directory(NOISE_UPLOAD_FOLDER, filename)

@app.route('/noise/result/<filename>')
def noise_result_file(filename):
    return send_from_directory(NOISE_RESULT_FOLDER, filename)

@app.route('/restore/upload/<filename>')
def restore_uploaded_file(filename):
    return send_from_directory(RESTORE_UPLOAD_FOLDER, filename)

@app.route('/restore/result/<filename>')
def restore_result_file(filename):
    return send_from_directory(RESTORE_RESULT_FOLDER, filename)

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    if folder == 'noise':
        folder_path = NOISE_RESULT_FOLDER
    elif folder == 'restore':
        folder_path = RESTORE_RESULT_FOLDER
    else:
        return "Folder tidak valid", 400

    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        return "File tidak ditemukan", 404
    
if __name__ == '__main__':
    app.run(debug=True)
