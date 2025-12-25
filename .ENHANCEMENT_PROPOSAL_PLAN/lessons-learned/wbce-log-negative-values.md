# WBCE Log Negative Values

**Keywords**: WBCE, log, negative, loss, numerical stability, smoothing
**Related Commits**: 97b4210

## Problem

In the Weighted Binary Cross-Entropy (WBCE) loss function, when a model is well-trained and predictions approach 1.0, adding smoothing (`y_pred + smooth`) can result in values exceeding 1.0. Taking `K.log()` of values greater than 1.0 produces positive results instead of negative, causing incorrect loss computation.

```python
# Problematic code
positive_loss = - weights * y_true * K.log(y_pred + smooth)
```

## Resolution

Clamp the log argument to a maximum of 1.0 using `K.minimum()` to ensure the log always receives valid probability values:

```python
# Fixed code
positive_loss = - weights * y_true * K.log(K.minimum(y_pred + smooth, 1))
negative_loss = - (1 - weights) * (1 - y_true) * K.log(K.minimum(1 - y_pred + smooth, 1))
```

## Prevention

- Always validate that log arguments remain in valid range `(0, 1]` for probability-based losses
- When adding smoothing constants, consider clamping the result
- Add unit tests with well-trained model outputs (predictions near 0 or 1) to catch numerical edge cases
