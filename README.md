# Full Resolution Image Compression with Recurrent Neural Networks
https://arxiv.org/abs/1608.05148v2

## Requirements
- PyTorch 0.2.0

## Train
`
python train.py -f /path/to/your/images/folder/like/mscoco
`

## Encode and Decode
### Encode
`
python encoder.py --model checkpoint/encoder_epoch_00000005.pth --input /path/to/your/example.png --cuda --output ex --iterations 16
`

This will output binary codes saved in `.npz` format.

### Decode
`
python decoder.py --model checkpoint/encoder_epoch_00000005.pth --input /path/to/your/example.npz --cuda --output /path/to/output/folder
`

This will output images of different quality levels.

## Test
### Get Kodak dataset
```bash
bash test/get_kodak.sh
```

### Demo
```bash
bash test/enc_dec.sh
```

## Official Repo
https://github.com/tensorflow/models/tree/master/compression
