import os
import argparse

import numpy as np
from matplotlib.image import imsave

import torch
from torch.autograd import Variable
import network


def decode(image_input, ouput_dir, model, n_iterations, cuda):
    content = np.load(image_input)
    codes = np.unpackbits(content["codes"])
    codes = np.reshape(codes, content["shape"]).astype(np.float32) * 2 - 1

    codes = torch.from_numpy(codes)
    iters, batch_size, channels, height, width = codes.size()
    height = height * 16
    width = width * 16
    codes = Variable(codes)

    decoder = network.DecoderCell()
    decoder.eval()

    decoder.load_state_dict(torch.load(model))

    decoder_h_1 = (
        Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
        Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ),
    )
    decoder_h_2 = (
        Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
        Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8),
        ),
    )
    decoder_h_3 = (
        Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
        Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4),
        ),
    )
    decoder_h_4 = (
        Variable(torch.zeros(batch_size, 128, height // 2, width // 2)),
        Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2),
        ),
    )

    if cuda:
        decoder = decoder.cuda()

        codes = codes.cuda()

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    image = torch.zeros(1, 3, height, width) + 0.5
    for iters in range(min(n_iterations, codes.size(0))):

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4
        )
        image = image + output.data.cpu()

        if not os.path.exists(ouput_dir):
            os.mkdir(ouput_dir)

        imsave(
            os.path.join(ouput_dir, "{:02d}.png".format(iters + 1)),
            np.squeeze(image.numpy().clip(0, 1) * 255.0)
            .astype(np.uint8)
            .transpose(1, 2, 0),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="path to model")
    parser.add_argument("--input", required=True, type=str, help="input codes")
    parser.add_argument("--output", default=".", type=str, help="output folder")
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument("--iterations", type=int, default=16, help="unroll iterations")
    args = parser.parse_args()

    with torch.no_grad():
        decode(args.input, args.output, args.model, args.iterations, args.cuda)
