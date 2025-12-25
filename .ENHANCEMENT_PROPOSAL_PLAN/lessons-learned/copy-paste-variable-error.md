# Copy-Paste Variable Error

**Keywords**: copy-paste, variable, typo, wrong variable, min, max
**Related Commits**: 2e34b02

## Problem

When copying similar code blocks, a variable name was not updated properly. In the `min` modality branch, the output was incorrectly assigned to `max_mosaic` instead of `min_mosaic`:

```python
# Problematic code
if 'min' in modality:
    min_mosaic = Lambda(lambda x: K.min(x, axis=1))(mosaic)
    output = max_mosaic  # Bug: should be min_mosaic
```

## Resolution

Ensure variable names are correctly updated when copying code:

```python
# Fixed code
if 'min' in modality:
    min_mosaic = Lambda(lambda x: K.min(x, axis=1))(mosaic)
    output = min_mosaic
```

## Prevention

- Review copy-pasted code carefully for variable name consistency
- Use IDE refactoring tools instead of manual copy-paste
- Add unit tests that verify output corresponds to the correct operation
- Consider using linting tools that detect unused variable assignments
- When adding new modalities, test each one individually
