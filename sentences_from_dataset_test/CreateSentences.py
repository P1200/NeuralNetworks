from PIL import Image
import numpy as np
import PolLettDS.PolLettDS as pld


def crop_letter_image(img: np.ndarray, threshold=10):
    """
    Przytnij czarne marginesy z lewej i prawej strony białej litery (na czarnym tle).
    Zostaw oryginalną wysokość.
    """
    mask = img > threshold  # biała litera na czarnym tle
    if not mask.any():
        return Image.fromarray(img)  # pusty obraz
    x_coords = np.where(mask.any(axis=0))[0]
    x0, x1 = x_coords[0], x_coords[-1] + 1
    cropped = img[:, x0:x1]  # tylko kolumny
    return Image.fromarray(cropped)

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    '../PolLettDS/pol_lett_ds.bin',
    '../PolLettDS/pol_lett_ds_labels.bin')

# Map labels to indices
unique_labels = np.unique(loaded_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in loaded_labels])

# --- 2. Przygotowanie ---
img_size = 64
block_size = img_size * img_size  # 4096

# Przekształć dane na array i podziel na osobne obrazki
loaded_data = np.array(loaded_data)
num_letters = loaded_data.size // block_size
split_images = loaded_data[:num_letters * block_size].reshape((num_letters, img_size, img_size))

# Mapa klas
classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

# --- 3. Mapowanie znaków na obrazki ---
set_number = 20  # <-- numer zestawu, od 0 w górę
num_classes = len(classes)
start_index = set_number * num_classes
end_index = start_index + num_classes

char_to_img = {}
for i in range(start_index, min(end_index, len(mapped_labels))):
    label = mapped_labels[i]
    char = classes[label]
    if char not in char_to_img:
        image_data = split_images[i]
        cropped_img = crop_letter_image(image_data.astype(np.uint8))
        char_to_img[char] = cropped_img

# --- 4. Tekst do narysowania ---
text = """Litwo! Ojczyzno moja! ty jesteś jak zdrowie.
Ile cię trzeba cenić, ten tylko się dowie,
Kto cię stracił. Dziś piękność twą w całej ozdobie
Widzę i opisuję, bo tęsknię po tobie."""

lines = text.split('\n')
space_size = img_size // 2

# --- 5. Składanie obrazka z wiersza ---
def render_line(line):
    images = []
    for char in line:
        if char == ' ':
            images.append(Image.new('L', (space_size, img_size), color=0))
        elif char in char_to_img:
            images.append(char_to_img[char])
        # Pomijamy brakujące znaki
    if not images:
        return Image.new('L', (1, img_size), color=0)  # pusty wiersz
    total_width = sum(im.width for im in images)
    line_img = Image.new('L', (total_width, img_size), color=0)
    x_offset = 0
    for im in images:
        line_img.paste(im, (x_offset, 0))
        x_offset += im.width
    return line_img

# --- 6. Łączenie wszystkich wierszy ---
line_images = [render_line(line) for line in lines]
final_width = max(img.width for img in line_images)
final_height = img_size * len(line_images)

final_image = Image.new('L', (final_width, final_height), color=0)
for idx, line_img in enumerate(line_images):
    final_image.paste(line_img, (0, idx * img_size))

# --- 7. Wyświetlenie lub zapis ---
# final_image.show()
final_image.save("pan_tadeusz.png")
