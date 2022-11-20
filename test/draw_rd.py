import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from metric import msssim

graph_path = "graphs"
images_path = "test/images"
decoded_path = "test/decoded"
jpeg_path = "test/jpeg"
LSTM_SSIM_CSV = "test/lstm_ssim.csv"
JPEG_SSIM_CSV = "test/jpeg_ssim.csv"
LSTM_BPP_CSV = "test/lstm_bpp.csv"
JPEG_BPP_CSV = "test/jpeg_bpp.csv"

lstm_bpp = []
lstm_ssim = []
jpeg_bpp = []
jpeg_ssim = []


def draw_graph(
    lstm_bpp, lstm_ssim, jpeg_bpp, jpeg_ssim, line=True, original_image=None
):
    jpeg_ssim = np.array(jpeg_ssim)
    jpeg_bpp = np.array(jpeg_bpp)
    lstm_ssim = np.array(lstm_ssim)
    lstm_bpp = np.array(lstm_bpp)

    if jpeg_ssim.ndim == 1:
        jpeg_ssim = np.expand_dims(jpeg_ssim, axis=0)
    if jpeg_bpp.ndim == 1:
        jpeg_bpp = np.expand_dims(jpeg_bpp, axis=0)
    if lstm_ssim.ndim == 1:
        lstm_ssim = np.expand_dims(lstm_ssim, axis=0)
    if lstm_bpp.ndim == 1:
        lstm_bpp = np.expand_dims(lstm_bpp, axis=0)

    if original_image:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        im_array = np.array(Image.open(original_image))
        ax1.imshow(im_array)
        ax1.axis("off")
        graph_name = os.path.basename(original_image)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 4))
        graph_name = "average"
    if line:
        ax0.plot(
            lstm_bpp.mean(axis=0), lstm_ssim.mean(axis=0), label="LSTM", marker="o"
        )
    else:
        ax0.scatter(
            lstm_bpp.reshape(-1), lstm_ssim.reshape(-1), label="LSTM", marker="o"
        )

    if line:
        ax0.plot(
            jpeg_bpp.mean(axis=0), jpeg_ssim.mean(axis=0), label="LSTM", marker="o"
        )
    else:
        ax0.scatter(
            jpeg_bpp.reshape(-1), jpeg_ssim.reshape(-1), label="JPEG", marker="x"
        )

    ax0.set_xlim(0.0, 2.0)
    ax0.set_ylim(0.68, 1.02)
    ax0.set_xlabel("bit per pixel")
    ax0.set_ylabel("MS-SSIM")
    ax0.legend()
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    fig.savefig(f"graphs/{graph_name}.png")


if __name__ == "__main__":
    for fname in tqdm(os.listdir(images_path)):
        if ":Zone.Identifier" in fname.lower():
            continue
        name, ext = fname.rsplit(".", 1)
        original_image = os.path.join(images_path, fname)

        decoded_folder = os.path.join(decoded_path, name)
        jpeg_folder = os.path.join(jpeg_path, name)

        print(f"Evaluating LSTM performance for {fname}")
        lstm_ssim_row = []
        lstm_bpp_row = []
        for i, decoded_name in enumerate(sorted(os.listdir(decoded_folder))):
            decoded_image = os.path.join(decoded_folder, decoded_name)
            score = msssim(original_image, decoded_image, crop_original=True)
            lstm_ssim_row.append(score)

            im = Image.open(original_image)
            n_frames = getattr(im, "n_frames", 1)
            h, w = im.size
            n_pixels = h * w // 3

            bits_with_iteration = (i + 1) / 192
            size_in_bits = (
                bits_with_iteration * os.path.getsize(decoded_image) / n_frames
            )

            bpp = size_in_bits / n_pixels
            lstm_bpp_row.append(bpp)
            # print(f"original_image: {original_image}, decoded_image: {decoded_image}, score: {score}, bpp: {bpp}")
        lstm_bpp.append(lstm_bpp_row)
        lstm_ssim.append(lstm_ssim_row)

        print(f"Evaluating JPEG performance for {fname}")
        jpeg_ssim_row = []
        jpeg_bpp_row = []
        for jpeg_name in sorted(os.listdir(jpeg_folder)):
            jpeg_image = os.path.join(jpeg_folder, jpeg_name)
            score = msssim(original_image, jpeg_image, crop_original=True)

            jpeg_ssim_row.append(score)
            size_in_bits = os.path.getsize(jpeg_image)
            h, w = Image.open(jpeg_image).size
            n_pixels = h * w // 3

            bpp = size_in_bits / n_pixels
            jpeg_bpp_row.append(bpp)
        jpeg_bpp.append(jpeg_bpp_row)
        jpeg_ssim.append(jpeg_ssim_row)

        draw_graph(
            lstm_bpp_row,
            lstm_ssim_row,
            jpeg_bpp_row,
            jpeg_ssim_row,
            line=True,
            original_image=original_image,
        )

    np.savetxt(LSTM_SSIM_CSV, lstm_ssim, delimiter=",")
    np.savetxt(JPEG_SSIM_CSV, jpeg_ssim, delimiter=",")
    np.savetxt(LSTM_BPP_CSV, lstm_bpp, delimiter=",")
    np.savetxt(JPEG_BPP_CSV, jpeg_bpp, delimiter=",")

    draw_graph(lstm_bpp, lstm_ssim, jpeg_bpp, jpeg_ssim, line=True)
