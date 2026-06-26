import torch


def add_integrated_gradients_to_trace(
    model,
    image_tensor: torch.Tensor,
    trace: dict,
    max_tokens: int = 12,
    steps: int = 16,
):
    tokens = trace.get('tokens', None)
    if tokens is None:
        return trace
    with torch.enable_grad():
        ig_attr = model.integrated_gradient_token_attributions(
            image_tensor,
            tokens,
            max_tokens=max_tokens,
            steps=steps,
        )
    trace['ig_attributions'] = ig_attr
    return trace
