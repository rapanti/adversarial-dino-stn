import torch
import torch.nn as nn
import torch.nn.functional as f


class SinkhornTopK(nn.Module):
    def __init__(self, mu, nu, epsilon, max_iter):
        super().__init__()
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.max_iter = max_iter

    def __call__(self, x):
        return TopKFunc.apply(x, self.mu, self.nu, self.epsilon, self.max_iter)


def sinkhorn_forward(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    device = C.device

    v = torch.ones([bs, 1, k_]).to(device) / (k_)
    G = torch.exp(-C / epsilon).to(device)

    for i in range(max_iter):
        u = mu / (G * v).sum(-1, keepdim=True)
        v = nu / (G * u).sum(-2, keepdim=True)

    Gamma = u * G * v
    return Gamma


def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):
    bs, n, k_ = C.size()
    device = C.device
    k = k_ - 1

    f = torch.zeros([bs, n, 1], device=device)
    g = torch.zeros([bs, 1, k + 1], device=device)

    epsilon_log_mu = epsilon * torch.log(mu)
    epsilon_log_nu = epsilon * torch.log(nu)

    def min_epsilon_row(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

    def min_epsilon_col(Z, epsilon):
        return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

    for i in range(max_iter):
        f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

    Gamma = torch.exp((-C + f + g) / epsilon)
    return Gamma


def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):
    nu_ = nu[:, :, :-1]
    Gamma_ = Gamma[:, :, :-1]

    bs, n, k_ = Gamma.size()

    inv_mu = 1. / (mu.view([1, -1]))  # [1, n]
    Kappa = torch.diag_embed(nu_.squeeze(-2)) \
            - torch.matmul(Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_)  # [bs, k, k]

    inv_Kappa = torch.inverse(Kappa)  # [bs, k, k]

    Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
    L = Gamma_mu.matmul(inv_Kappa)  # [bs, n, k]
    G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

    g1 = G1.sum(-1)
    G21 = (g1 * inv_mu).unsqueeze(-1) * Gamma  # [bs, n, k+1]
    g1_L = g1.unsqueeze(-2).matmul(L)  # [bs, 1, k]
    G22 = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma  # [bs, n, k+1]
    G23 = - f.pad(g1_L, pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G2 = G21 + G22 + G23  # [bs, n, k+1]

    del g1, G21, G22, G23, Gamma_mu

    g2 = G1.sum(-2).unsqueeze(-1)  # [bs, k+1, 1]
    g2 = g2[:, :-1, :]  # [bs, k, 1]
    G31 = - L.matmul(g2) * Gamma  # [bs, n, k+1]
    G32 = f.pad(inv_Kappa.matmul(g2).transpose(-1, -2), pad=(0, 1), mode='constant', value=0) * Gamma  # [bs, n, k+1]
    G3 = G31 + G32  # [bs, n, k+1]

    grad_C = (-G1 + G2 + G3) / epsilon  # [bs, n, k+1]
    return grad_C


class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = sinkhorn_forward(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print('Nan appeared in Gamma, re-computing...')
                    Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [bs, n, k+1]
        with torch.no_grad():
            grad_C = sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)
        return grad_C, None, None, None, None


class TopK_custom(torch.nn.Module):
    def __init__(self, k, epsilon=0.001, max_iter=500):
        super().__init__()
        self.k = k
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1])
        self.max_iter = max_iter

    def forward(self, scores):
        bs, n = scores.size()
        device = scores.device
        scores = scores.view([bs, n, 1])

        # find the -inf value and replace it with the minimum value except -inf
        scores_ = scores.clone().detach()
        max_scores = torch.max(scores_).detach()
        scores_[scores_ == float('-inf')] = float('inf')
        min_scores = torch.min(scores_).detach()
        filled_value = min_scores - (max_scores - min_scores)
        mask = scores == float('-inf')
        scores = scores.masked_fill(mask, filled_value)

        C = (scores - self.anchors.to(device)) ** 2
        C = C / (C.max().detach())

        mu = torch.ones([1, n, 1], requires_grad=False, device=device) / n
        nu = [1. / n for _ in range(self.k)]
        nu.append((n - self.k) / n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k + 1]).to(device)

        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)

        A = Gamma[:, :, :self.k] * n

        return A
