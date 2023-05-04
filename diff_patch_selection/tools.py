import torch
import torch.nn.functional as f
import einops


def extract_images_patches(images,
                           window_size,
                           stride=1):
    """Extracts patches from an image using a convolution operator.

    Args:
    images: A tensor of images of shapes (B, C, H, W).
    window_size: The size of the patches to extract (h, w).
    stride: The shift between extracted patches (s1, s2)

    Returns:
    All the patches in a tensor of dimension
      (B, (H - h + 1) // s1, (W - w + 1) // s2, C, h, w).
    """
    bs, c, height, width = images.shape
    h, w = window_size

    concatenated_patches = f.unfold(images, kernel_size=(h, w), stride=stride)

    patches = einops.rearrange(concatenated_patches, "b (c h w) n -> b n c h w", c=c, h=h, w=w)
    return patches


def extract_images_patches_old(images,
                               window_size,
                               stride=1):
    """Extracts patches from an image using a convolution operator.

    Args:
    images: A tensor of images of shapes (B, C, H, W).
    window_size: The size of the patches to extract (h, w).
    stride: The shift between extracted patches (s1, s2)

    Returns:
    All the patches in a tensor of dimension
      (B, (H - h + 1) // s1, (W - w + 1) // s2, h, w, C).
    """
    d = images.shape[1]
    h, w = window_size

    # construct the lookup conv weights
    dim_out = torch.arange(d * h * w).reshape((-1, 1, 1, 1))
    dim_in = torch.arange(d).reshape((1, -1, 1, 1))
    i = torch.arange(h).reshape((1, 1, -1, 1))
    j = torch.arange(w).reshape((1, 1, 1, -1))
    weights = ((w * i + j) * d + dim_in == dim_out).type(images.dtype).to(images.device)

    # batch, h * w * d, (H - h + 1) // s1, (W - w + 1) // s2
    concatenated_patches = f.conv2d(images, weights, stride=stride)
    print("concatenated_patches.shape", concatenated_patches.shape)

    # batch, (H - h + 1) // s1, (W - w + 1) // s2, h * w * d
    concatenated_patches = torch.moveaxis(concatenated_patches, 1, -1)

    # batch, (H - h + 1) // s1, (W - w + 1) // s2, h, w, d
    shape = concatenated_patches.shape[:3] + (h, w, d)
    patches = concatenated_patches.reshape(shape)
    return patches
