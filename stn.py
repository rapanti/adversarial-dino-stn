import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from utils import grad_reverse


N_PARAMS = {
    'affine': 6,
    'rotation': 1,
    'rotation_scale': 3,
    'rotation_scale_symmetric': 2,
    'rotation_translation': 3,
    'rotation_translation_scale': 5,
    'rotation_translation_scale_symmetric': 4,
    'scale': 2,
    'scale_symmetric': 1,
    'translation': 2,
    'translation_scale': 4,
    'translation_scale_symmetric': 3,
}

IDENT_TENSORS = {
    'affine': [1, 0, 0, 0, 1, 0],
    'rotation': [0],
    'rotation_scale': [0, 1, 1],
    'rotation_scale_symmetric': [0, 1],
    'rotation_translation': [0, 0, 0],
    'rotation_translation_scale': [0, 0, 0, 1, 1],
    'rotation_translation_scale_symmetric': [0, 0, 0, 1],
    'scale': [1, 1],
    'scale_symmetric': [1],
    'translation': [0, 0],
    'translation_scale': [0, 0, 1, 1],
    'translation_scale_symmetric': [0, 0, 1],
}


class LocBackbone(nn.Module):
    def __init__(self, in_depth=32, out_depth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, in_depth, 3, padding=2)
        self.bn1 = nn.BatchNorm2d(in_depth)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_depth, out_depth, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(out_depth)
        self.pool2 = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x


class LocHead(nn.Module):
    def __init__(self, mode, feature_dim: int):
        super().__init__()
        self.mode = mode
        self.stn_n_params = N_PARAMS[mode]
        self.feature_dim = feature_dim
        self.linear0 = nn.Linear(feature_dim, 128)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, self.stn_n_params)

        # Initialize the weights/bias with identity transformation
        self.linear2.weight.data.zero_()
        self.linear2.bias.data.copy_(torch.tensor(IDENT_TENSORS[mode], dtype=torch.float))

    def forward(self, x):
        xs = torch.flatten(x, 1)
        xs = F.leaky_relu(self.linear0(xs))
        xs = F.leaky_relu(self.linear1(xs))
        xs = self.linear2(xs)
        return xs


class LocNet(nn.Module):
    """
    Localization Network for the Spatial Transformer Network. Consists of a ResNet-Backbone and FC-Head
    """

    def __init__(self, mode: str = 'affine', invert_gradient: bool = False,
                 num_heads: int = 4, separate_backbones: bool = False,
                 conv1: int = 32, conv2: int = 32):
        super().__init__()
        self.mode = mode
        self.invert_gradient = invert_gradient
        self.separate_backbones = separate_backbones
        self.num_heads = num_heads
        self.feature_dim = conv2 * 8 ** 2

        num_backbones = num_heads if self.separate_backbones else 1

        self.backbones = nn.ModuleList(
            [LocBackbone(conv1, conv2) for _ in range(num_backbones)]
        )
        self.heads = nn.ModuleList(
            [LocHead(self.mode, self.feature_dim) for _ in range(self.num_heads)]
        )

    def forward(self, x):
        if self.separate_backbones:
            outputs = [h(b(x)) for b, h in zip(self.backbones, self.heads)]
        else:
            xs = self.backbones[0](x)
            outputs = [head(xs) for head in self.heads]
        if self.invert_gradient:
            outputs = [grad_reverse(theta) for theta in outputs]
        return outputs


class STN(nn.Module):
    """
    Spatial Transformer Network with a ResNet localization backbone
    """
    def __init__(self, mode: str = 'affine', invert_gradients: bool = False,
                 local_crops_number: int = 2,
                 separate_localization_net: bool = False,
                 conv1_depth: int = 32, conv2_depth: int = 32,
                 unbounded_stn: bool = False,
                 theta_norm: bool = False,
                 resolution: tuple = (224, 96),
                 global_crops_scale: tuple = (0.4, 1), local_crops_scale: tuple = (0.05, 0.4),):
        super().__init__()
        self.mode = mode
        self.stn_n_params = N_PARAMS[mode]
        self.separate_localization_net = separate_localization_net
        self.invert_gradients = invert_gradients
        self.conv1_depth = conv1_depth
        self.conv2_depth = conv2_depth
        self.theta_norm = theta_norm
        self.unbounded_stn = unbounded_stn
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        assert len(resolution) in (1, 2), f"resolution parameter should be of length 1 or 2, but {len(resolution)} with {resolution} is given."
        self.global_res, self.local_res = resolution[0] + resolution[0] if len(resolution) == 1 else resolution

        self.total_crops_number = 2 + self.local_crops_number
        # Spatial transformer localization-network
        self.localization_net = LocNet(self.mode, self.invert_gradients, self.total_crops_number,
                                       self.separate_localization_net, self.conv1_depth, self.conv2_depth)

    def _get_stn_mode_theta(self, theta, x):  # Fastest
        if self.mode == 'affine':
            theta = theta if self.unbounded_stn else torch.tanh(theta)
            return theta.view(-1, 2, 3)

        out = torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device(), requires_grad=True) + 0
        a, b, tx = [1., 0., 0.]
        c, d, ty = [0., 1., 0.]

        if 'rotation' in self.mode:
            angle = theta[:, 0]
            a = torch.cos(angle)
            b = -torch.sin(angle)
            c = torch.sin(angle)
            d = torch.cos(angle)
        if 'translation' in self.mode:
            x, y = (0, 1) if self.mode.startswith('translation') else (1, 2)
            tx = theta[:, x] if self.unbounded_stn else torch.tanh(theta[:, x])
            ty = theta[:, y] if self.unbounded_stn else torch.tanh(theta[:, y])
        if 'scale' in self.mode:
            if 'symmetric' in self.mode:
                sx = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
                sy = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
            else:
                sx = theta[:, -2] if self.unbounded_stn else torch.tanh(theta[:, -2])
                sy = theta[:, -1] if self.unbounded_stn else torch.tanh(theta[:, -1])
            a *= sx
            b *= sx
            c *= sy
            d *= sy

        out[:, 0, 0] = a
        out[:, 0, 1] = b
        out[:, 0, 2] = tx
        out[:, 1, 0] = c
        out[:, 1, 1] = d
        out[:, 1, 2] = ty
        return out

    def forward(self, x):
        theta_params = self.localization_net(x)

        thetas = [self._get_stn_mode_theta(params, x) for params in theta_params]

        if self.theta_norm:
            thetas = [theta / theta.norm(1, dim=2, keepdim=True).clamp(min=1).max(dim=1, keepdim=True).values
                      for theta in thetas]
            # normed = []
            # for theta in thetas:
            #     norm = theta.norm(1, dim=2, keepdim=True).norm(2, dim=1, keepdim=True).clamp(min=1).repeat(1, 1, 3)
            #     norm[:, :, -1] = 1.
            #     theta = theta / norm
            #     normed.append(theta)
            # thetas = normed

        align_corners = True
        crops = []
        grids = []
        resolutions = [[self.global_res, self.global_res]] * 2 + \
                      [[self.local_res, self.local_res]] * self.local_crops_number
        for theta, res in zip(thetas, resolutions):
            grid = F.affine_grid(theta, size=list(x.size()[:2]) + res, align_corners=align_corners)
            grids.append(grid)
            crop = F.grid_sample(x, grid, align_corners=align_corners)
            crops.append(crop)

        return crops, thetas, grids


class AugmentationNetwork(nn.Module):
    def __init__(self, transform_net: STN, resize_input: bool = False, resize_size: int = 512):
        super().__init__()
        print("Initializing Augmentation Network")
        self.transform_net = transform_net
        self.resize_input = resize_input
        self.resize_size = resize_size

    def forward(self, x):
        # if we get a tensor as input, simply pass it to the STN
        if isinstance(x, torch.Tensor):
            return self.transform_net(x)

        # otherwise the input should be a list of PIL images, e.g. uncropped ImageNet dataset
        if not isinstance(x, list):
            x = [x]
        for idx, img in enumerate(x):
            if self.resize_input and max(img.size()) > self.resize_size:
                img = resize(img, size=[self.resize_size, ], max_size=self.resize_size+1)
            img = img.unsqueeze(0)
            x[idx] = img

        num_crops = self.transform_net.local_crops_number + 2
        views = [[] for _ in range(num_crops)]
        thetas = [[] for _ in range(num_crops)]
        for img in x:
            views_net, thetas_net = self.transform_net(img)

            for idx, (view, theta) in enumerate(zip(views_net, thetas_net)):
                views[idx].append(view)
                thetas[idx].append(theta)

        views = [torch.cat(view) for view in views]
        thetas = [torch.cat(theta) for theta in thetas]

        return views, thetas
