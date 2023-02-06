import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import affine_grid
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms
from utils import grad_reverse, grad_rescale


class ThetaCropsPenalty(nn.Module):
    def __init__(self,
                 invert: bool = False,
                 eps: float = 1.,
                 local_crops_scale=(0.05, 0.4),
                 global_crops_scale=(0.4, 1),
                 loss_fn=nn.HuberLoss,
                 **kwargs):
        super().__init__()
        self.invert = invert
        self.eps = eps
        self.loss_fn = loss_fn()
        self.local_crops_scale = local_crops_scale
        self.global_crops_scale = global_crops_scale

    def forward(self, thetas, **args):
        loss = 0
        for x in thetas[:2]:
            loss += self._loss(x, self.global_crops_scale)
        for x in thetas[2:]:
            loss += self._loss(x, self.local_crops_scale)

        loss /= len(thetas)

        if self.invert:
            loss = grad_reverse(loss, self.eps)
        else:
            loss = grad_rescale(loss, self.eps)

        return loss

    def _loss(self, theta, scale):
        a, b = scale
        targed = (b - a) * torch.rand(theta.size(0), device=theta.get_device()) + a
        # targed = (a + b) / 2

        # narrow area down for the translation parameters, reduce adversarial freedom
        # stay closer to the center [0, 0]
        low = math.pow(a, 0.125)
        high = math.pow(b, 0.125)
        # sample uniformly in range [low, high]
        target = (high - low) * torch.rand(theta.size(0), device=theta.get_device()) + low
        # target = (low + high) / 2

        det = torch.det(theta[:, :, :2].float()).abs()
        txy = (1 - (theta[:, :, 2].abs() / 2)).prod(dim=1)

        return self.loss_fn(det, targed) + self.loss_fn(txy, target)


class OverlapPenalty(nn.Module):
    """
    Another penalty that uses theta. Generates the sampling/affine grid from theta. The grid is then used to calculate
    the overlap between the different views. Global to global and local to local cross-similarity.

    Two different approaches are implemented:
    1. _affine_grids + _overlap1: affine_grids works as expected, overlap1 calculates then the overlap based on the
        minima and maxima per row, column and  axes
    2. _flatten_grids + _overlap: similar to affine grids but flattened per axes, therefore the minima and maxima
        are calculated per axes.
    """
    def __init__(self, eps=1, invert=False, **kwargs):
        super().__init__()
        self.eps = eps
        self.invert = invert

    def forward(self, thetas, **args):
        # create own affine grids of thetas
        # grids = self._affine_grids(thetas) working with self._overlap1
        grids = self._flatten_grids(thetas)

        total_loss = 0
        n_loss_terms = 0

        # global views
        global1, global2 = grids[:2]
        # loss = self._overlap1(global1, global2) working with _affine_grids
        loss = self._overlap(global1, global2)
        total_loss += loss.mean()
        n_loss_terms += 1

        # local views
        for ida, a in enumerate(grids[2:]):
            for idb in range(ida + 3, len(grids)):
                b = grids[idb]
                # loss = self._overlap1(a, b) working with _affine_grids
                loss = self._overlap(a, b)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        if self.invert:
            total_loss = grad_reverse(total_loss, self.eps)
        else:
            total_loss = grad_rescale(total_loss, self.eps)

        return total_loss

    @staticmethod
    def _affine_grids(thetas):
        """
        Generally affine grid produces grids with shape: [BS, W, H, 2]
        Here 2x2 affine grids should be sufficient: [BS, 2, 2, 2]
        """
        grids = [affine_grid(theta, [theta.size(0), 1, 2, 2], True) for theta in thetas]
        return grids

    @staticmethod
    def _flatten_grids(thetas):
        """
        Flattened and reshaped grid: [BS, 2, 4], instead of [BS, 2, 2, 2]
        """
        grids = []
        for theta in thetas:
            grid = affine_grid(theta, [theta.size(0), 1, 2, 2], True)  # grid.shape: [bs, 2, 2, 2]
            grid = grid.reshape(grid.size(0), 2, -1)  # grid.shape: [bs, 2, 4]
            grids.append(grid)
        return grids

    @staticmethod
    def _overlap(grid1, grid2):
        left1, right1 = grid1[:, 0, :].min(dim=1).values, grid1[:, 0, :].max(dim=1).values  # min/max of x-coord/axis
        top1, bottom1 = grid1[:, 1, :].min(dim=1).values, grid1[:, 1, :].max(dim=1).values  # min/max of y-coord/axis
        left2, right2 = grid2[:, 0, :].min(dim=1).values, grid2[:, 0, :].max(dim=1).values  # min/max of x-coord/axis
        top2, bottom2 = grid2[:, 1, :].min(dim=1).values, grid2[:, 1, :].max(dim=1).values  # min/max of y-coord/axis

        x = torch.clamp(torch.min(right1, right2) - torch.max(left1, left2), 0)  # length of x-overlap
        y = torch.clamp(torch.min(bottom1, bottom2) - torch.max(top1, top2), 0)  # length of y-overlap
        area = x * y / 4.

        return area

    @staticmethod
    def _overlap1(grid1, grid2):
        left1, top1 = grid1[:, :, 0, 0].min(dim=-1).values, grid1[:, 0, :, 0].min(dim=-1).values
        right1, bottom1 = grid1[:, :, -1, 0].max(dim=-1).values, grid1[:, -1, :, 0].max(dim=-1).values
        left2, top2 = grid2[:, :, 0, 0].min(dim=-1).values, grid2[:, 0, :, 0].min(dim=-1).values
        right2, bottom2 = grid2[:, :, -1, 0].max(dim=-1).values, grid2[:, -1, :, 0].max(dim=-1).values

        x = torch.clamp(torch.min(right1, right2) - torch.max(left1, left2), 0)
        y = torch.clamp(torch.min(bottom1, bottom2) - torch.max(top1, top2), 0)
        area = x * y / 4.

        return area

    @staticmethod
    def _overlap2(grid1, grid2):
        """NOT WORKING - NO GRADIENT"""
        left1, right1 = grid1[:, 0, 0, 0], grid1[:, -1, -1, 0]
        bottom1, top1 = grid1[:, -1, -1, 1], grid1[:, 0, 0, 1]
        left2, right2 = grid2[:, 0, 0, 0], grid2[:, -1, -1, 0]
        bottom2, top2 = grid2[:, -1, -1, 1], grid2[:, 0, 0, 1]

        x = torch.clamp(torch.min(right1, right2) - torch.max(left1, left2), 0)
        y = torch.clamp(torch.min(bottom1, bottom2) - torch.max(top1, top2), 0)

        return x * y / 4.


class ThetaLoss(nn.Module):
    def __init__(self, eps=1, invert=False, **kwargs):
        super().__init__()
        self.invert = invert
        self.eps = eps
        self.loss_fn = nn.MSELoss()

    def forward(self, thetas, **args):
        identity = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=thetas[0].get_device())
        loss = 0
        for t in thetas:
            loss += self.loss_fn(t, identity)

        loss /= len(thetas)

        if self.invert:
            loss = grad_reverse(loss, self.eps)
        else:
            loss = grad_rescale(loss, self.eps)

        return loss


class SIMLoss(nn.Module):
    def __init__(self, resolution: int, min_sim: float = 1., invert=False, eps: float = 1., **kwargs):
        super().__init__()
        self.loss_fn = SSIM()
        self.resize = transforms.Resize(resolution)
        self.min_sim = 1 - min_sim
        self.invert = invert
        self.eps = eps

    def forward(self, images, target, **kwargs):
        target = self.resize(torch.stack(target))
        total_loss = 0
        for img in images:
            loss = 1 - self.loss_fn(self.resize(img), target)
            loss[loss < self.min_sim] = 0
            total_loss += loss

        total_loss /= len(images)

        if self.invert:
            total_loss = grad_reverse(total_loss, self.eps)
        else:
            total_loss = grad_rescale(total_loss, self.eps)

        return total_loss


"""
###################################
LEGACY CODE ||| SAVED FOR LATER USE
###################################
"""


class GridLoss(nn.Module):
    """
    Actually not needed. It is the same as the ThetaLoss.
    """

    def __init__(self, device=torch.device('cuda'), **kwargs):
        super().__init__()
        self.identity = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float, device=device)
        self.loss_fn = nn.MSELoss()

    def forward(self, grid, **kwargs):
        loss = 0
        for g in grid:
            size = [g.size(0), 1, g.size(1), g.size(2)]
            identity = self.identity.expand(g.size(0), 2, 3)
            grid_identity = F.affine_grid(identity, size)
            loss = loss + self.loss_fn(g, grid_identity)
        return loss


def histogram_batch(
        input: torch.Tensor,
        bins: int = 30,
        min: float = 0.0,
        max: float = 1.0,
        kernel: str = "gaussian"
) -> torch.Tensor:
    """Estimates the histogram of the input.
    The calculation uses kernel density estimation. Default 'epanechnikov' kernel.

    Args:
        input: Input tensor to compute the histogram with shape :math:`(B, d1, d2, ...)`
        bins: The number of histogram bins.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive).
        kernel: kernel to perform kernel density estimation
          ``(`epanechnikov`, `gaussian`, `triangular`, `uniform`)``.
    Returns:
        Computed histogram of shape :math:`(B, bins)`.
    """
    if input is not None and not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not torch.Tensor. Got {type(input)}.")
    if not isinstance(bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(bins)}.")
    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')
    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=input.device, dtype=input.dtype) + 0.5)
    centers = centers.reshape(-1, 1, 1)
    u = torch.abs(input.flatten(1).unsqueeze(0) - centers) / delta

    if kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u ** 2)
    elif kernel in ("triangular", "uniform", "epanechnikov"):
        # compute the mask and cast to floating point
        mask = (u <= 1).to(u.dtype)
        if kernel == "triangular":
            kernel_values = (1.0 - u) * mask
        elif kernel == "uniform":
            kernel_values = torch.ones_like(u) * mask
        else:  # kernel == "epanechnikov"
            kernel_values = (1.0 - u ** 2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', 'uniform' or 'epanechnikov'. Got {kernel}.")
    hist = torch.sum(kernel_values, dim=-1).permute(1, 0)
    return hist


def histogram(
        input: torch.Tensor,
        bins: int = 30,
        min: float = 0.0,
        max: float = 1.0,
        kernel: str = "gaussian"
) -> torch.Tensor:
    """Estimates the histogram of the input.
    The calculation uses kernel density estimation. Default 'epanechnikov' kernel.

    Args:
        input: Input tensor to compute the histogram with shape :math:`(d1, d2, ...)`
        bins: The number of histogram bins.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive).
        kernel: kernel to perform kernel density estimation
          ``(`epanechnikov`, `gaussian`, `triangular`, `uniform`)``.
    Returns:
        Computed histogram of shape :math:`(B, bins)`.
    """
    if input is not None and not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not torch.Tensor. Got {type(input)}.")
    if not isinstance(bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(bins)}.")
    if not isinstance(min, float):
        raise TypeError(f'Type of lower end of the range is not a float. Got {type(min)}.')
    if not isinstance(max, float):
        raise TypeError(f"Type of upper end of the range is not a float. Got {type(min)}.")

    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=input.device, dtype=torch.half) + 0.5)
    centers = centers.reshape(-1, 1)
    u = torch.abs(input.flatten().unsqueeze(0) - centers) / delta
    # creates a (B x bins x (3 x H x W))-shape tensor

    if kernel == "gaussian":
        kernel_values = torch.exp(-0.5 * u ** 2)
    elif kernel in ("triangular", "uniform", "epanechnikov"):
        # compute the mask and cast to floating point
        mask = (u <= 1).to(u.dtype)
        if kernel == "triangular":
            kernel_values = (1.0 - u) * mask
        elif kernel == "uniform":
            kernel_values = torch.ones_like(u) * mask
        else:  # kernel == "epanechnikov"
            kernel_values = (1.0 - u ** 2) * mask
    else:
        raise ValueError(f"Kernel must be 'triangular', 'gaussian', 'uniform' or 'epanechnikov'. Got {kernel}.")

    hist = torch.sum(kernel_values, dim=-1)
    return hist


class HSIM(nn.Module):
    def __init__(self, exponent=1):
        super(HSIM, self).__init__()
        self.exponent = exponent

    def forward(self, pred, target):
        t = histogram_batch(target)
        p = histogram_batch(pred)
        m = torch.min(p, t)
        mask = (p == 0).to(p.dtype)
        p = p + mask
        score = torch.sum(torch.pow(m / p, self.exponent)) / t.shape.numel()
        return score


class HISTLoss(nn.Module):
    def __init__(self, bins=100, exponent=2, invert=False, **kwargs):
        super().__init__()
        self.bins = bins
        self.exponent = exponent
        self.invert = -1 if invert else 1
        self.fn = HSIM()

    def forward(self, input, target):
        if isinstance(target, list):
            target = torch.stack(target)
        loss = 0
        for crop in input:
            score = self.fn(crop, target)
            step = 1 - score
            loss += step
        return self.invert * loss
