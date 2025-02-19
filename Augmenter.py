import cv2
import numpy as np


class Augmenter:
    def __init__(self):
        pass

    def augment_dataset(self, images, labels, shift_pixels=5, rotation_degrees=1, scale_factor=0.01):
        h, w = images[0].shape[:2]

        augmented_images = []
        augmented_labels = []

        for image, label in zip(images, labels):
            augmented_images.append(image)  # Dodaj oryginalny obraz
            augmented_labels.append(label)

            # Przesunięcie, obrót i skalowanie dla różnych wariantów
            for angle in [-rotation_degrees, 0, rotation_degrees]:
                for dx in [-shift_pixels, 0, shift_pixels]:
                    for dy in [-shift_pixels, 0, shift_pixels]:
                        for scale in [1 - scale_factor, 1, 1 + scale_factor]:  # Skalowanie ±10%
                            if dx == 0 and dy == 0 and angle == 0 and scale == 1:
                                continue  # Pomijamy oryginalny obrazek

                            # Przesunięcie
                            M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
                            translated_image = cv2.warpAffine(image, M_translate, (w, h), borderMode=cv2.BORDER_REFLECT)

                            # Obrót + Skalowanie
                            M_rotate_scale = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
                            transformed_image = cv2.warpAffine(translated_image, M_rotate_scale, (w, h),
                                                               borderMode=cv2.BORDER_REFLECT)

                            augmented_images.append(transformed_image)
                            augmented_labels.append(label)

        # Konwersja listy do tablicy NumPy (znacznie szybsze niż append)
        return np.array(augmented_images), np.array(augmented_labels)
