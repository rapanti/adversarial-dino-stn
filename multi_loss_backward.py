import torch


class MultiLossBack(object):
    def __init__(self, params, max_norm=1.):
        super(MultiLossBack).__init__()
        self.params = params
        self.max_norm = max_norm
        self.params = list(params)

    def backward(self, loss_array, weights):
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            self._clip_gradients()
            for p in self.params:
                if p.grad is None:
                    break

                if p.grad.is_sparse:
                    raise RuntimeError('MLB does not support sparse gradients.')

                if loss_index == 0:
                    p.grads = p.grad.detach() * weights[loss_index]
                else:
                    p.grads += p.grad.detach() * weights[loss_index]

                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

        for p in self.params:
            p.grad = p.grads

    def _clip_gradients(self):
        grads = [p.grad for p in self.params if p.grad is not None]
        device = grads[0].device
        total_norm = torch.stack([p.detach().data.norm(2) for p in grads]).data.norm(2).to(device)
        clip_coef = self.max_norm / (total_norm + 1e-5)
        for g in grads:
            g.detach().mul_(clip_coef)

