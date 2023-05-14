import enum

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from diff_patch_selection import TopK_custom
from diff_patch_selection import PerturbedTopK
from diff_patch_selection import tools


class SelectionMethod(str, enum.Enum):
    SINKHORN_TOPK = "topk"
    PERTURBED_TOPK = "perturbed-topk"
    HARD_TOPK = "hard-topk"
    RANDOM = "random"


class SqueezeExciteLayer(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = num_channels // reduction
        self.model = nn.Sequential(
            nn.Linear(num_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = self.model(y)
        return x * y[:, :, None, None]


class Scorer(nn.Module):
    def __init__(self, use_excite=False, pool_size=4, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(inplace=True),
            SqueezeExciteLayer(16) if use_excite else nn.Identity(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            SqueezeExciteLayer(32) if use_excite else nn.Identity(),
            nn.Conv2d(32, 1, 3),
            nn.MaxPool2d(pool_size, pool_size),
        )

    def forward(self, x):
        return self.model(x)

    @classmethod
    def compute_output_size(cls, height, width, pool_size):
        return (height - 8) // pool_size, (width - 8) // pool_size


class PatchNet(nn.Module):
    def __init__(self,
                 patch_size,
                 k,
                 use_scorer_se,
                 selection_method,
                 normalization_str,
                 hard_topk_probability,
                 random_patch_probability):
        super().__init__()
        self.patch_size = patch_size
        self.k = k
        self.use_scorer_se = use_scorer_se
        self.selection_method = SelectionMethod(selection_method)
        self.extract_by_indices = self.selection_method in [SelectionMethod.HARD_TOPK,
                                                            SelectionMethod.RANDOM]
        self.hard_topk_probability = hard_topk_probability
        self.random_patch_probability = random_patch_probability
        self.scorer = Scorer()
        self.topk_fn = TopK_custom(self.k) if self.selection_method is SelectionMethod.SINKHORN_TOPK else None

        self.norm_fn = create_normalization_fn(normalization_str)

    def forward(self, x):
        bs, c, h, w = x.shape

        low = h // 4
        mid = 2 * h // 3
        # Randomize patch size
        patch_size = np.random.choice(range(low, mid), self.k)

        # === Compute the scores ===
        if self.selection_method == SelectionMethod.RANDOM:
            scores_h, scores_w = Scorer.compute_output_size(h, w)
            num_patches = scores_h * scores_w
        else:
            scores = self.scorer(x)
            flatten_scores = scores.flatten(1)
            num_patches = flatten_scores.size(-1)
            scores_h, scores_w = scores.shape[-2:]

            # prob_scores = f.softmax(flatten_scores, dim=-1)
            flatten_scores = self.norm_fn(flatten_scores)
            # scores = flatten_scores.reshape(scores.shape)

        # === Patch Selection ===
        if self.selection_method is SelectionMethod.SINKHORN_TOPK:
            indicators = self.topk_fn(flatten_scores)
            indicators = einops.rearrange(indicators, "b n k -> b k n")
        elif self.selection_method is SelectionMethod.PERTURBED_TOPK:
            topk_fn = PerturbedTopK(self.k)
            indicators = topk_fn(flatten_scores)
        elif self.selection_method is SelectionMethod.HARD_TOPK:
            indices = select_patches_hard_topk(flatten_scores, self.k)
        elif self.selection_method is SelectionMethod.RANDOM:
            rows = [torch.multinomial(input=torch.ones(num_patches), num_samples=self.k, replacement=False) for _ in
                    range(bs)]
            indices = torch.stack(rows).to(x.device)

        indicators = zerooneeps(indicators.exp().exp().exp())
        indicators = indicators / indicators.sum(dim=-1, keepdims=True)

        # Randomly use hard topk at training
        if self.training and self.hard_topk_probability > 0 and \
                self.selection_method not in [SelectionMethod.HARD_TOPK, SelectionMethod.RANDOM]:
            true_indices = select_patches_hard_topk(flatten_scores, self.k)
            random_values = torch.rand(bs, device=x.device)
            use_hard = random_values < self.hard_topk_probability
            if self.extract_by_indices:
                indices = torch.where(use_hard[:, None], true_indices, indices)
            else:
                true_indicators = f.one_hot(true_indices, num_classes=num_patches).float()
                indicators = torch.where(use_hard[:, None, None], true_indicators, indicators)

        # Sample some random patches during training with random_patch_probability.
        if self.training and self.random_patch_probability > 0 and \
                self.selection_method is not SelectionMethod.RANDOM:
            rows = [torch.multinomial(input=torch.ones(num_patches), num_samples=self.k, replacement=False) for _ in
                    range(bs)]
            random_indices = torch.stack(rows).to(x.device)
            random_values = torch.rand(bs, self.k)
            use_random = random_values < self.random_patch_probability
            if self.extract_by_indices:
                indices = torch.where(use_random, random_indices, indices)
            else:
                random_indicators = f.one_hot(random_indices, num_classes=num_patches).float()
                indicators = torch.where(use_random[:, None, :], random_indicators, indicators)

        # === Patch extraction ===
        if self.extract_by_indices:
            patches = extract_patches_from_indices(
                x, indices,
                patch_size=self.patch_size, grid_shape=(scores_h, scores_w))
        else:
            patches = extract_patches_from_indicators(
                x, indicators,
                patch_size, grid_shape=(scores_h, scores_w))
        return patches


def select_patches_hard_topk(scores, k):
    return torch.argsort(scores)[:, -k:]


def calculate_stride_pad(h, w, patch_size, scores_h, scores_w):
    stride_h = round((h - patch_size) / (scores_h - 1))
    stride_w = round((w - patch_size) / (scores_w - 1))
    padding = stride_h * (scores_h - 1) + patch_size - h
    pad_l = padding // 2 if padding else 0
    pad_r = padding % 2 + pad_l if padding else 0
    return stride_h, stride_w, pad_l, pad_r


def extract_patches_from_indicators(x, indicators, patch_size, grid_shape):
    bs, c, h, w = x.shape
    scores_h, scores_w = grid_shape
    chunks = indicators.size(1)
    patches = []
    for indicator, ps in zip(indicators.chunk(chunks, dim=1), patch_size):
        print(ps)
        stride_h, stride_w, pad_h, pad_w = calculate_stride_pad(h, w, ps, scores_h, scores_w)
        print(stride_h, stride_w, pad_h, pad_w)
        padded_x = f.pad(x, (pad_h, pad_w, pad_h, pad_w))
        patch = tools.extract_images_patches(
            padded_x,
            window_size=(ps, ps),
            stride=(stride_h, stride_w)
        )
        print(patch.shape)
        patch = torch.einsum("b k n, b n c i j -> b k c i j", indicator, patch)
        patches.append(patch)

    return patches


def extract_patches_from_indices(x, indices, patch_size, grid_shape):
    bs, c, h, w = x.shape
    scores_h, scores_w = grid_shape
    num_patches = scores_h * scores_w
    indices = f.one_hot(indices, num_classes=num_patches).float()

    stride_h, stride_w, pad_h, pad_w = calculate_stride_pad(h, w, patch_size, scores_h, scores_w)
    padded_x = f.pad(x, (pad_h, pad_w, pad_h, pad_w))
    patches = tools.extract_images_patches(
        padded_x,
        window_size=(patch_size, patch_size),
        stride=(stride_h, stride_w)
    )
    patches = torch.einsum("b k n, b n c i j -> b k c i j", indices, patches)

    return patches


def _get_available_normalization_fns():
    def smoothing(s):
        def smoothing_fn(x):
            uniform = 1. / x.size(-1)
            x = x * (1 - s) + uniform * s
            return x

        return smoothing_fn

    def zeroone(scores):
        scores -= scores.min(dim=-1, keepdim=True).values
        scores /= scores.max(dim=-1, keepdim=True).values
        return scores

    def zerooneeps(eps):
        def zerooneeps_fn(scores):
            scores_min = scores.min(dim=-1, keepdim=True).values
            scores_max = scores.max(dim=-1, keepdim=True).values
            return (scores - scores_min) / (scores_max - scores_min + eps)

        return zerooneeps_fn

    return dict(
        identity=lambda x: x,
        softmax=nn.Softmax(dim=1),
        smoothing=smoothing,
        zeroone=zeroone,
        zerooneeps=zerooneeps,
        sigmoid=nn.Sigmoid(),
        exp=torch.exp,
    )


def zerooneeps(scores, eps=1e-5):
    scores_min = scores.min(dim=-1, keepdim=True).values
    scores_max = scores.max(dim=-1, keepdim=True).values
    return (scores - scores_min) / (scores_max - scores_min + eps)


def create_normalization_fn(fn_str):
    functions = [eval(fn, _get_available_normalization_fns())
                 for fn in fn_str.split("|") if fn]

    def chain(x):
        for fn in functions:
            x = fn(x)
        return x

    return chain


if __name__ == "__main__":
    from torchvision import transforms, datasets
    from tests.plot_scripts import plot
    import matplotlib.pyplot as plt

    toPIL = transforms.ToPILImage()
    cifar = datasets.CIFAR10("../../../datasets/cifar10", transform=transforms.ToTensor())

    image, _ = cifar.__getitem__(11)
    x = image.unsqueeze(0)

    patch_net = PatchNet(16, 4, False, "perturbed-topk", "exp|exp|zerooneeps(1e-4)", 0., 0)

    out = patch_net(x)

    # chunked_out = out.chunk(6, dim=1)
    images = [toPIL(image)] + [toPIL(img.squeeze()) for img in out]

    plot(images)
    plt.show()

    loss_fn = nn.MSELoss()
    target = torch.zeros_like(out)

    patch_net.zero_grad()

    loss = loss_fn(target, out)
    loss.backward()

    print(loss)
