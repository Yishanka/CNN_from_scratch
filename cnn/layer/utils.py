from cnn.core import Tensor
def im2col(x: Tensor, kernel_size, stride, padding) -> Tensor:
    # x: [batch_size, in_channels, height, weight]
    _, _, h, w = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding

    # padding
    if ph > 0 or pw > 0:
        x = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))  # [bs, ic, h+2ph, w+2pw]

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    cols = []
    for i in range(oh):
        for j in range(ow):
            patch = x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]  # [bs, ic, kh, kw]
            cols.append(patch.reshape((x.shape[0], -1)))   # [bs, ic*kh*kw]
    return Tensor.stack(cols, axis=1)  # [bs, oh*ow, ic*kh*kw]