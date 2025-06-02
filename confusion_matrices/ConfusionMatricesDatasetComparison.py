import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.metrics import structural_similarity as ssim
import PolLettDB.PolLettDB as pld
import matplotlib.patheffects as path_effects


def compute_confusion_matrix(images):
    n = len(images)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = np.nan  # same picture
            elif i < j:
                # euclidean distance
                dist = np.linalg.norm(images[i].astype(np.float32) - images[j].astype(np.float32))
                matrix[i, j] = dist
            else:
                # SSIM
                ssim_val, _ = ssim(images[i], images[j], full=True)
                matrix[i, j] = ssim_val

    return matrix


def save_confusion_matrix(matrix, filename='confusion_matrix_dual.png'):
    n = matrix.shape[0]

    upper = np.triu(matrix, k=1)
    lower = np.tril(matrix, k=-1)

    # Change NaN to 0 and force float32
    upper_disp = np.where(np.isnan(upper), 0, upper).astype(np.float32)
    lower_disp = np.where(np.isnan(lower), 0, lower).astype(np.float32)

    fig, ax = plt.subplots(figsize=(25, 25))

    im1 = ax.imshow(upper_disp, cmap='Reds', interpolation='none')
    im2 = ax.imshow(lower_disp, cmap='Blues', interpolation='none')

    # Set alpha masks as float32
    alpha_upper = np.where(upper_disp > 0, 1.0, 0.0).astype(np.float32)
    alpha_lower = np.where(lower_disp > 0, 1.0, 0.0).astype(np.float32)

    im1.set_alpha(alpha_upper)
    im2.set_alpha(alpha_lower)

    # ax.set_xticks([])
    # ax.set_yticks([])

    tick_labels = ['0'] * n
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tick_labels, fontsize=25)
    ax.set_yticklabels(tick_labels, fontsize=25)



    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, 'eq', ha='center', va='center', color='black', fontsize=28)

                # text = ax.text(j, i, 'eq',
                #                ha='center', va='center', color='black', fontsize=28)
                #
                # text.set_path_effects([
                #     path_effects.Stroke(linewidth=3, foreground='white'),
                #     path_effects.Normal()
                # ])
            elif i < j:
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha='center', va='center', color='black', fontsize=28)

                    # text = ax.text(j, i, f"{val:.0f}",
                    #                ha='center', va='center', color='black', fontsize=28)
                    #
                    # text.set_path_effects([
                    #     path_effects.Stroke(linewidth=3, foreground='white'),
                    #     path_effects.Normal()
                    # ])
            else:
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha='center', va='center', color='white', fontsize=28)

                    # text = ax.text(j, i, f"{val:.3f}",
                    #                ha='center', va='center', color='white', fontsize=28)
                    #
                    # text.set_path_effects([
                    #     path_effects.Stroke(linewidth=3, foreground='black'),
                    #     path_effects.Normal()
                    # ])

    ax.set_title(f'Macierz konfuzji dla znaku {classes[labels[0]]} ze zbioru danych: Odległość (czerwony) nad przekątną, SSIM (niebieski) pod przekątną',
                 fontsize=30)

    # cbar1 = fig.colorbar(im1, ax=ax, fraction=0.026, pad=0.06)
    # cbar1.set_label('Odległość euklidesowa', rotation=270, labelpad=35, fontsize=25)
    # cbar1.ax.tick_params(labelsize=20)
    #
    # cbar2 = fig.colorbar(im2, ax=ax, fraction=0.026, pad=0.02)
    # cbar2.set_label('SSIM', rotation=270, labelpad=35, fontsize=25)
    # cbar2.ax.tick_params(labelsize=20)

    # Colorbar 1 – Odległość euklidesowa (czerwony)
    cax1 = inset_axes(ax, width="80%", height="2.5%", loc='lower center',
                      bbox_to_anchor=(0.1, -0.05, 0.8, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    cb1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cb1.set_label('Odległość euklidesowa', fontsize=30)
    cb1.ax.tick_params(labelsize=25)

    # Colorbar 2 – SSIM (niebieski)
    cax2 = inset_axes(ax, width="80%", height="2.5%", loc='lower center',
                      bbox_to_anchor=(0.1, -0.15, 0.8, 1),
                      bbox_transform=ax.transAxes, borderpad=0)
    cb2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
    cb2.set_label('SSIM', fontsize=30)
    cb2.ax.tick_params(labelsize=25)

    # plt.xlabel('Etykieta osi X')
    # plt.ylabel('Etykieta osi Y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Macierz zapisana do {filename}")


loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    '../PolLettDB/pol_lett_db.bin',
    '../PolLettDB/pol_lett_db_labels.bin')

# Map labels to indices
unique_labels = np.unique(loaded_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in loaded_labels])

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

block_size = 64 * 64
count = labels_count * block_size

for d in range(0, 80):

    a = block_size * d + block_size * 80 * 11  # start point
    chars = []
    labels = []

    for i in range(a, count // 2, block_size * 80):
        chars.append(loaded_data[i:i + block_size].reshape(64, 64))
        labels.append(mapped_labels[i // block_size])

    # plt.figure(figsize=(4, 10))
    # for i, (img, label) in enumerate(zip(chars[:80], labels[:80])):
    #     plt.subplot(17, 5, i + 1)
    #     plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #     plt.axis('off')
    #     plt.text(32, 70, str(label), fontsize=8, ha='center', va='top')
    #
    # plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)
    # plt.tight_layout(pad=0.2)
    # plt.show()

    conf_matrix = compute_confusion_matrix(chars)
    save_confusion_matrix(conf_matrix,
                          filename=f'same_chars_comparison/{d:02d}_confusion_matrix_{classes[labels[0]]}.png')
