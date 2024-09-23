class AWP:
    def __init__(self, model, criterion, optimizer, adv_param="weight", adv_lr=1e-3, adv_eps=1e-2):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.adv_lr * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)
                    param.data = torch.clamp(param.data, self.backup[name] - self.adv_eps, self.backup[name] + self.adv_eps)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                param.data = self.backup[name]
        self.backup = {}

    def attack_backward(self, low_resolution_image, high_resolution_image):
        self.perturb()
        self.optimizer.zero_grad()
        output = self.model(low_resolution_image)
        loss = self.criterion(output, high_resolution_image)
        loss.backward()
        self.restore()
