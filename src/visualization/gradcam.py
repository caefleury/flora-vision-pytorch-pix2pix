import torch
import torch.nn.functional as F
import numpy as np
import cv2


# classe base para metodos CAM
class GradCAMBase:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()

    def _compute_cam(self, weights):
        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * self.activations, dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()

    def generate(self, input_tensor, target_output=None):
        raise NotImplementedError


class GradCAM(GradCAMBase):

    def generate(self, input_tensor, target_output=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_output is None:
            target = output.mean()
        else:
            target = (output * target_output).sum()

        target.backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3)).squeeze(0)
        return self._compute_cam(weights)


class GradCAMPlusPlus(GradCAMBase):

    def generate(self, input_tensor, target_output=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_output is None:
            target = output.mean()
        else:
            target = (output * target_output).sum()

        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        grad_2 = gradients**2
        grad_3 = gradients**3
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3)).squeeze(0)
        return self._compute_cam(weights)


class ScoreCAM(GradCAMBase):

    def generate(self, input_tensor, target_output=None, batch_size=16):
        with torch.no_grad():
            _ = self.model(input_tensor)

        activations = self.activations
        b, c, h, w = activations.shape

        input_h, input_w = input_tensor.shape[2:]
        upsampled = F.interpolate(
            activations, size=(input_h, input_w), mode="bilinear", align_corners=False
        )

        upsampled = upsampled.reshape(b, c, -1)
        min_vals = upsampled.min(dim=2, keepdim=True)[0]
        max_vals = upsampled.max(dim=2, keepdim=True)[0]
        upsampled = (upsampled - min_vals) / (max_vals - min_vals + 1e-8)
        upsampled = upsampled.reshape(b, c, input_h, input_w)

        scores = []

        with torch.no_grad():
            for i in range(0, c, batch_size):
                batch_masks = upsampled[:, i : i + batch_size]
                masked_inputs = input_tensor * batch_masks

                for j in range(batch_masks.shape[1]):
                    output = self.model(masked_inputs[:, j : j + 1].expand_as(input_tensor))
                    score = output.mean().item()
                    scores.append(score)

        scores = torch.tensor(scores)
        scores = F.softmax(scores, dim=0)

        cam = (scores.view(1, -1, 1, 1).to(activations.device) * activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


class EigenCAM(GradCAMBase):

    def generate(self, input_tensor, target_output=None):
        with torch.no_grad():
            _ = self.model(input_tensor)

        activations = self.activations.squeeze(0)
        c, h, w = activations.shape

        activations_2d = activations.reshape(c, -1)

        U, S, V = torch.svd(activations_2d.float())

        projection = torch.matmul(S[0:1], V[:, 0:1].t())
        cam = projection.reshape(h, w)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()


class LayerCAM(GradCAMBase):

    def generate(self, input_tensor, target_output=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_output is None:
            target = output.mean()
        else:
            target = (output * target_output).sum()

        target.backward(retain_graph=True)

        cam = F.relu(self.gradients * self.activations).sum(dim=1)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
    "layercam": LayerCAM,
}


def get_available_cam_methods():
    return list(CAM_METHODS.keys())


def get_cam_method_description(method):
    descriptions = {
        "gradcam": "Grad-CAM: Usa gradientes para ponderar mapas de ativacao. Metodo padrao.",
        "gradcam++": "Grad-CAM++: Versao melhorada com ponderacao pixel a pixel.",
        "scorecam": "Score-CAM: Sem gradientes, usa perturbacoes. Mais lento mas estavel.",
        "eigencam": "Eigen-CAM: Usa componentes principais. Rapido e sem gradientes.",
        "layercam": "Layer-CAM: Multiplicacao de gradientes e ativacoes. Boa localizacao.",
    }
    return descriptions.get(method, "Metodo desconhecido")


class ColorModelGradCAM:

    def __init__(self, model, method="gradcam"):
        self.model = model
        self.method = method
        self.generator = self._get_generator()
        self.target_layers = self._get_target_layers()
        self.cam_objects = {}

    def _get_generator(self):
        if hasattr(self.model, "netG"):
            return self.model.netG
        elif hasattr(self.model, "model"):
            if hasattr(self.model.model, "netG"):
                return self.model.model.netG
        raise ValueError("Nao foi possivel encontrar rede geradora no modelo")

    def _get_target_layers(self):
        layers = {}

        if hasattr(self.generator, "model"):
            model = self.generator.model

            def find_layers(module, prefix=""):
                for name, child in module.named_children():
                    full_name = f"{prefix}.{name}" if prefix else name

                    if isinstance(child, torch.nn.Conv2d):
                        layers[full_name] = child
                    elif isinstance(child, torch.nn.ConvTranspose2d):
                        layers[full_name] = child
                    elif isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList)):
                        find_layers(child, full_name)
                    else:
                        find_layers(child, full_name)

            find_layers(model)

        if not layers:
            for name, module in self.generator.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    layers[name] = module

        return layers

    def get_available_layers(self):
        return list(self.target_layers.keys())

    def generate_cam(self, input_tensor, layer_name):
        if layer_name not in self.target_layers:
            raise ValueError(f"Camada '{layer_name}' nao encontrada. Disponiveis: {self.get_available_layers()}")

        target_layer = self.target_layers[layer_name]
        cam_class = CAM_METHODS.get(self.method, GradCAM)

        cam = cam_class(self.generator, target_layer)

        try:
            heatmap = cam.generate(input_tensor)
            input_h, input_w = input_tensor.shape[2:]
            heatmap_resized = cv2.resize(heatmap, (input_w, input_h))
            return heatmap_resized
        finally:
            cam.remove_hooks()

    def generate_multi_layer_cam(self, input_tensor, layer_names=None):
        if layer_names is None:
            all_layers = self.get_available_layers()
            n_layers = min(5, len(all_layers))
            indices = np.linspace(0, len(all_layers) - 1, n_layers, dtype=int)
            layer_names = [all_layers[i] for i in indices]

        results = {}
        for layer_name in layer_names:
            try:
                results[layer_name] = self.generate_cam(input_tensor, layer_name)
            except Exception as e:
                print(f"Aviso: Nao foi possivel gerar CAM para camada '{layer_name}': {e}")

        return results


def apply_colormap(heatmap, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_cam_on_image(image, cam, alpha=0.5):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    cam_colored = apply_colormap(cam)

    overlay = (1 - alpha) * image + alpha * cam_colored
    return np.clip(overlay, 0, 255).astype(np.uint8)
