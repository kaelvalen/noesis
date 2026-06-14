# NOESIS Backbone Patches

This directory contains optional patches for backbone model repositories so
that NOESIS can extract pre-output hidden states efficiently, avoiding the
heuristic inverse-mapping used in `NoesisBackbone._approximate_hidden_from_logits()`.

## Why

NOESIS needs **both** the next-token logits (for generation) and the pre-output
hidden states `h_t` (for TTT and Titans). The RWKV reference implementation's
`forward()` returns only `(logits, state)`. Running `forward()` twice — once for
logits and once for hidden states — is wasteful and risks state drift.

## RWKV Patch

Apply to `rwkv/model.py` in the `rwkv` package.

### Option A: Modify `forward()` to return hidden states

In the `RWKV.forward()` method, after the final hidden state `x` is computed
but before the output head, capture `x` and return it alongside logits:

```python
# Before
return logits, state

# After
return logits, state, x  # x: (batch*seq, d_model)
```

Then update `NoesisBackbone.forward_with_hidden()` to unpack three values when
`self._backend_name == "rwkv"`.

### Option B: Add a `return_hidden` flag

A non-breaking alternative:

```python
def forward(self, tokens, state=None, return_hidden=False):
    ...
    if return_hidden:
        return logits, state, x
    return logits, state
```

Then set `return_hidden=True` in `NoesisBackbone.forward_with_hidden()`.

## Mamba Patch

`mamba_ssm` already exposes hidden states through the model layers in many
versions. If your version does not, patch `MambaLMHeadModel.forward()` to
return `last_hidden_state` in addition to logits.

## Note

Without these patches, NOESIS falls back to a logits→hidden approximation
that is sufficient for testing but suboptimal for production.
