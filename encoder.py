import argparse

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import network


def encode(input_image, npz_output, model, n_iterations, cuda):
    _image = Image.open(input_image).convert(mode="RGB")
    _image = np.array(_image)
    _height, _weight = (_image.shape[0] // 32) * 32, (_image.shape[1] // 32) * 32
    _image = _image[:_height, :_weight, :]

    image = torch.from_numpy(
        np.expand_dims(
            np.transpose(np.array(_image).astype(np.float32) / 255.0, (2, 0, 1)), 0
        )
    )

    batch_size, input_channels, height, width = image.size()
    assert height % 32 == 0 and width % 32 == 0

    encoder = network.EncoderCell()
    binarizer = network.Binarizer()
    decoder = network.DecoderCell()

    encoder.eval()
    binarizer.eval()
    decoder.eval()

    encoder.load_state_dict(torch.load(model))
    binarizer.load_state_dict(torch.load(model.replace("encoder", "binarizer")))
    decoder.load_state_dict(torch.load(model.replace("encoder", "decoder")))

    image = Variable(
        image,
    )
    encoder_h_1 = (
        Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4),
        ),
        Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4),
        ),
    )
    encoder_h_2 = (
        Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8),
        ),
        Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8),
        ),
    )
    encoder_h_3 = (
        Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ),
        Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ),
    )

    decoder_h_1 = (
        Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ),
        Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16),
        ),
    )
    decoder_h_2 = (
        Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8),
        ),
        Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8),
        ),
    )
    decoder_h_3 = (
        Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4),
        ),
        Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4),
        ),
    )
    decoder_h_4 = (
        Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2),
        ),
        Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2),
        ),
    )

    if cuda:
        encoder = encoder.cuda()
        binarizer = binarizer.cuda()
        decoder = decoder.cuda()

        image = image.cuda()

        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    codes = []
    res = image - 0.5
    for iters in range(n_iterations):
        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3
        )

        code = binarizer(encoded)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4
        )

        res = res - output
        codes.append(code.data.cpu().numpy())

        print("Iter: {:02d}; Loss: {:.06f}".format(iters + 1, res.data.abs().mean()))

    codes = (np.stack(codes).astype(np.int8) + 1) // 2

    export = np.packbits(codes.reshape(-1))

    np.savez_compressed(npz_output, shape=codes.shape, codes=export)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, type=str, help="path to model")
    parser.add_argument("--input", "-i", required=True, type=str, help="input image")
    parser.add_argument("--output", "-o", required=True, type=str, help="output codes")
    parser.add_argument("--cuda", "-g", action="store_true", help="enables cuda")
    parser.add_argument("--iterations", type=int, default=16, help="unroll iterations")
    args = parser.parse_args()

    with torch.no_grad():
        encode(args.input, args.output, args.model, args.iterations, args.cuda)
