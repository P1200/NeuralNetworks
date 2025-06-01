import os
import re

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from conv_network.ConvolutionalNN import ConvolutionalNN


def preprocess_letter(letter_image, target_size=64, max_letter_size=30):

    coords = cv2.findNonZero(letter_image)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = letter_image[y:y + h, x:x + w]

    scale = max_letter_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    final_image = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    final_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return final_image


# Upload model
model = ConvolutionalNN(num_classes=80)
model.load_state_dict(torch.load("../cnn_model.pth"))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Transform images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Read images
folder_path = "../sentences_from_dataset_test/literki"

image_files = sorted(
    [f for f in os.listdir(folder_path) if f.endswith('.png')],
    key=lambda x: int(re.search(r"(\d+)", x).group(1))
)

predicted_chars = []
letter_images = []

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    image = Image.open(img_path)
    image = np.array(image)

    image = preprocess_letter(image)
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_char = classes[predicted_class]
    predicted_chars.append(predicted_char)
    letter_images.append(image)

# Show results
num_letters = len(letter_images)
letter_size = 64
label_height = 30

letters_per_row = 10
num_rows = (num_letters + letters_per_row - 1) // letters_per_row

canvas_width = letters_per_row * letter_size
canvas_height = num_rows * (letter_size + label_height)

canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255  # białe tło

canvas_pil = Image.fromarray(canvas).convert("RGB")
draw = ImageDraw.Draw(canvas_pil)

try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

for idx, (img, label) in enumerate(zip(letter_images, predicted_chars)):
    row = idx // letters_per_row
    col = idx % letters_per_row

    x = col * letter_size
    y = row * (letter_size + label_height)

    letter_pil = Image.fromarray(img)
    canvas_pil.paste(letter_pil.convert("RGB"), (x, y))

    draw.text((x + 5, y + letter_size + 5), label, font=font, fill=(0, 0, 0))

plt.figure(figsize=(12, 8))
plt.imshow(canvas_pil, cmap='gray')
plt.axis('off')
plt.show()

# Or save to file
# cv2.imwrite("recognized_letters.png", canvas)

final_text = ''.join(predicted_chars)
print("Odczytany napis:", final_text)
