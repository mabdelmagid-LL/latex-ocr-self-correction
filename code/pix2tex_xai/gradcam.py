import torch


def add_gradcam_to_trace(model, image_tensor: torch.Tensor, trace: dict, max_tokens: int = 12):
    tokens = trace.get('tokens', None)
    if tokens is None:
        return trace
    with torch.enable_grad():
        grad_attr = model.gradient_token_attributions(image_tensor, tokens, max_tokens=max_tokens)
    trace['grad_attributions'] = grad_attr
    return trace
