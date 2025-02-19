import numpy as np
import PolLettDB.PolLettDB as pld
import cv2
import matplotlib.pyplot as plt


def augment_image_numpy_opencv(image_array, shift_pixels=5, rotation_degrees=10, scale_factor=0.1):
    h, w = image_array.shape[:2]
    augmented_images = []

    # Przesunięcie, obrót i skalowanie dla różnych wariantów
    for angle in [-rotation_degrees, 0, rotation_degrees]:
        for dx in [-shift_pixels, 0, shift_pixels]:
            for dy in [-shift_pixels, 0, shift_pixels]:
                for scale in [1 - scale_factor, 1, 1 + scale_factor]:  # Skalowanie ±10%
                    if dx == 0 and dy == 0 and angle == 0 and scale == 1:
                        continue  # Pomijamy oryginalny obrazek

                    # Przesunięcie
                    M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
                    translated_image = cv2.warpAffine(image_array, M_translate, (w, h), borderMode=cv2.BORDER_REFLECT)

                    # Obrót + Skalowanie
                    M_rotate_scale = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
                    transformed_image = cv2.warpAffine(translated_image, M_rotate_scale, (w, h),
                                                       borderMode=cv2.BORDER_REFLECT)

                    augmented_images.append(transformed_image)

    return augmented_images


loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    'PolLettDB/pol_lett_db.bin',
    'PolLettDB/pol_lett_db_labels.bin')

d=1202
block_size=64*64
a=64*64 * d
image_numpy = loaded_data[a:a+block_size].reshape(64,64)

# Augmentacja
augmented_images = augment_image_numpy_opencv(image_numpy, shift_pixels=5, rotation_degrees=10)

# Wizualizacja przykładowych wyników
plt.figure(figsize=(10, 5))
for i, img in enumerate(augmented_images[:27]):  # Wyświetl kilka obrazów
    plt.subplot(9, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show()
