# Division by Zero in Normalization

**Keywords**: division, zero, normalization, confusion matrix, ZeroDivisionError
**Related Commits**: c9f1b38

## Problem

In the confusion matrix normalization code, dividing by the row sum without checking if it's zero causes division by zero errors. This occurs when a class has no samples in the validation set.

```python
# Problematic code
for i in range(confusion_matrix.shape[0]):
    confusion_matrix[i] /= np.sum(confusion_matrix[i])
```

## Resolution

Check if the denominator is non-zero before performing division:

```python
# Fixed code
for i in range(confusion_matrix.shape[0]):
    line_sum = np.sum(confusion_matrix[i])
    if line_sum:
        confusion_matrix[i] = confusion_matrix[i] / line_sum
```

## Prevention

- Always guard division operations with zero checks, especially in data processing
- Consider using `np.divide(a, b, out=np.zeros_like(a), where=b!=0)` for vectorized safe division
- Add test cases with empty classes or edge case distributions
