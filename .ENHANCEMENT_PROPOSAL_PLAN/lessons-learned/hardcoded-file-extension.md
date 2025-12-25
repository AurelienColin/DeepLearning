# Hardcoded File Extension

**Keywords**: glob, file extension, png, npy, hardcoded, generator
**Related Commits**: 8bb9a64

## Problem

The segmenter generator only supported `.png` files because the glob pattern was hardcoded. This prevented using `.npy` (numpy arrays) or other file formats.

```python
# Problematic code
input_filenames = np.array(sorted(glob.glob(os.path.join(root, input_label, '*.png'))))
output_filenames = np.array(sorted(glob.glob(os.path.join(root, output_label, '*.png'))))
```

## Resolution

Use a wildcard pattern to match all file extensions:

```python
# Fixed code
input_filenames = np.array(sorted(glob.glob(os.path.join(root, input_label, '*.*'))))
output_filenames = np.array(sorted(glob.glob(os.path.join(root, output_label, '*.*'))))
```

## Prevention

- Avoid hardcoding file extensions unless there's a specific reason
- Use configurable extension parameters or wildcard patterns
- When changing supported formats, check all glob patterns in generators
- Document supported file formats in docstrings
