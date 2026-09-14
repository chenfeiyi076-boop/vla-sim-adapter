"""One strict Hybrid optimizer step; no distributed setup, loading or scheduling."""

import math

import torch


def hybrid_training_step(training_model, batch, normalizer, optimizer, *, device_type,
                         autocast_dtype=torch.bfloat16, autocast_enabled=True):
    """Enter the outer model/DDP once, validate every gradient, then step once.

    The caller supplies the validated Hybrid parameter policy and optimizer.
    `.module` is used only for inspection. Returns Python diagnostics only;
    gradients remain available for the caller's synchronization diagnostics.
    """
    module = getattr(training_model, "module", training_model)
    named = list(module.named_parameters())
    selected = {id(p) for _, p in named if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if not selected or len(actual) != len(selected) or set(actual) != selected:
        raise ValueError("Optimizer must contain exactly all selected trainable parameters, without duplicates/exclusions")
    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_enabled):
        loss, diagnostics = training_model(batch, normalizer)
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise ValueError("Local loss must be a finite scalar tensor")
    loss_value = float(loss.detach())
    if not math.isfinite(loss_value):
        raise ValueError("Local loss must be finite")
    loss.backward()
    missing = [name for name, p in named if p.requires_grad and p.grad is None]
    excluded = [name for name, p in named if not p.requires_grad and p.grad is not None]
    queries_ok = all(p.grad is None for p in module.encoder.action_queries.parameters())
    if missing or excluded or not queries_ok:
        raise RuntimeError(f"Gradient contract failed: missing_trainable_grads={missing}, "
                           f"excluded_with_grad={excluded}, action_queries_no_grad={queries_ok}")
    norms = []
    for group in (module.encoder, module.flow_head):
        gradients = [p.grad for p in group.parameters() if p.requires_grad]
        if not gradients:
            raise RuntimeError("Encoder and flow head must each have trainable gradients")
        # No per-parameter host synchronization. Non-finite elements make the
        # aggregate norm non-finite; a zero group norm also fails below.
        norms.append(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gradients]).norm())
    encoder_norm, flow_norm = torch.stack(norms).detach().cpu().tolist()
    if not all(math.isfinite(value) and value > 0 for value in (encoder_norm, flow_norm)):
        raise RuntimeError(f"Gradient norms must be finite and nonzero: encoder={encoder_norm}, flow_head={flow_norm}")
    result = dict(loss=loss_value, t_mean=float(diagnostics["t_mean"]),
                  velocity_rms=float(diagnostics["velocity_rms"]), batch_size=int(diagnostics["batch_size"]),
                  encoder_grad_norm=encoder_norm, flow_head_grad_norm=flow_norm,
                  missing_trainable_grads=[], excluded_no_grad=True, action_queries_no_grad=True)
    optimizer.step()
    return result
