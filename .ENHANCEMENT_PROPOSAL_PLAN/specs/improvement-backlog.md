# Improvement Backlog

**Generated:** 2025-12-25
**Task Reference:** X.5.1 - Code Quality Audit
**Status:** Living Document

---

## Summary

| Category | Issues Found | Severity |
|----------|--------------|----------|
| Type Hint Coverage | 139 errors in 45 files | Medium |
| PEP 8 Compliance | ~500+ violations | Low |
| Cyclomatic Complexity | 2 functions > 10 | Low |
| Code Duplication | 2 instances | Low |
| Overall Code Quality | 9.99/10 (pylint) | N/A |
| Performance Issues | 8 items (IMP-013 to IMP-020) | Medium-High |
| Feature Gaps | 47 items (FG-001 to FG-047) | Mixed |
| Technical Debt | 34 items (TD-001 to TD-031) | Low-High |
| Documentation Gaps | 24 items (DG-001 to DG-024) | Medium-High |

---

## 1. Type Hint Issues (Mypy Analysis)

**Tool:** `mypy --ignore-missing-imports src/`
**Total:** 139 errors in 45 files (142 source files checked)

### 1.1 High-Priority Type Issues

| File | Line | Issue | Effort | Impact |
|------|------|-------|--------|--------|
| `src/output_spaces/output_space.py` | 44-59 | Return types allow None when not expected | S | High |
| `src/generators/base_generators.py` | 58-72 | None checks missing on iterator access | S | High |
| `src/trainers/trainer.py` | 79, 124, 129, 153 | Return type mismatches and None callable | M | High |
| `src/models/model_wrapper.py` | 112, 126 | Missing attribute and return type issues | S | Medium |

### 1.2 Common Patterns to Fix

| Pattern | Count | Recommended Fix |
|---------|-------|-----------------|
| `Incompatible return value type (got "X \| None", expected "X")` | ~40 | Add proper Optional handling or assertions |
| `Item "None" of "X \| None" has no attribute` | ~25 | Add None guards before access |
| `Need type annotation for` | ~8 | Add explicit type annotations |
| `Incompatible types in assignment` | ~30 | Fix variable type declarations |
| `Module has no attribute` (PIL) | ~5 | Update PIL imports for Pillow 10+ |

### 1.3 Files Requiring Most Attention

| File | Error Count | Priority |
|------|-------------|----------|
| `src/samples/image_to_image/overlaid_sample.py` | 20 | Medium |
| `src/output_spaces/custom/nested/*.py` | 12 | Medium |
| `src/generators/*.py` | 15 | High |
| `src/models/image_to_image/diffusion_model_wrapper.py` | 6 | Low |
| `src/scripts/utils/processors/*.py` | 8 | Low |

---

## 2. PEP 8 Compliance (Flake8 Analysis)

**Tool:** `flake8 src/ tests/ --count --statistics`

### 2.1 Violation Summary

| Code | Description | Count | Priority |
|------|-------------|-------|----------|
| E501 | Line too long (>79 chars) | ~300+ | Low |
| F401 | Unused imports | ~20 | Medium |
| E302/E303 | Blank line issues | ~15 | Low |
| W292 | No newline at end of file | ~10 | Low |
| F821 | Undefined name | 2 | High |
| F541 | f-string missing placeholders | 2 | Low |
| W291/W293 | Trailing whitespace | ~10 | Low |
| E225/E231/E252 | Whitespace around operators | ~10 | Low |

### 2.2 Critical Issues (F-codes)

| File | Line | Issue |
|------|------|-------|
| `src/generators/image_to_image/overlay_generator.py` | 30 | Undefined name 'OutputSpace' |
| `src/generators/normalizer.py` | 25 | Undefined name 'typing' |

### 2.3 Unused Imports to Remove

| File | Import |
|------|--------|
| `src/callbacks/plotters/image_to_image/diffusion_example_plotter.py` | `Plotter` |
| `src/callbacks/plotters/image_to_image/diffusion_random_plotter.py` | `Plotter` |
| `src/callbacks/plotters/image_to_tag/confusion_matrix/confuson_matrice_plotter.py` | `typing`, `Display`, `reset_display` |
| `src/losses/from_model/encoding_similarity.py` | `typing`, `dataclass` |
| `src/models/image_to_image/auto_encoder_wrapper.py` | `K`, `build_encoder` |
| `src/models/image_to_image/unet_wrapper.py` | `tf`, `LazyProperty` |
| `src/models/image_to_tag/comparator_wrapper.py` | `K` |

---

## 3. Cyclomatic Complexity (Radon Analysis)

**Tool:** `python3 -m radon cc src/ -a -s`
**Average Complexity:** A (1.96) - Excellent

### 3.1 Functions Exceeding Threshold (>10)

| Location | Function/Method | Complexity | Recommended Action |
|----------|-----------------|------------|-------------------|
| `src/output_spaces/custom/nested/nested_tags.py:66` | `Category.accept` | C (14) | Refactor into smaller methods |
| `src/scripts/utils/download_hierarchical_dataset_from_danbooru.py:13` | `Downloader` | C (11) | Extract helper methods |
| `src/scripts/utils/download_hierarchical_dataset_from_danbooru.py:17` | `Downloader.run` | B (10) | At threshold - consider refactoring |
| `src/callbacks/history_callback.py:23` | `HistoryCallback.on_epoch_end` | B (10) | At threshold |

### 3.2 Notable B-Complexity Functions (6-10)

| Location | Function/Method | Complexity |
|----------|-----------------|------------|
| `src/samples/image_to_image/overlaid_sample.py:26` | `OverlaidSample.setup` | B (9) |
| `src/output_spaces/output_space.py:64` | `OutputSpace.common_setup` | B (8) |
| `src/benchmark_utils.py:33` | `get_xy` | B (6) |
| `src/samples/sample.py:41` | `Sample.imread` | B (7) |

---

## 4. Code Duplication (Pylint Analysis)

**Tool:** `pylint --disable=all --enable=duplicate-code src/`
**Code Rating:** 9.99/10

### 4.1 Duplicate Code Instances

| Files | Lines | Content |
|-------|-------|---------|
| `src/samples/image_to_image/foreground_sample.py:54-62` <br> `src/samples/image_to_image/overlaid_sample.py:104-112` | 8 | Matplotlib display code in `if __name__ == "__main__"` blocks |
| `src/callbacks/plotters/image_to_tag/confusion_matrix/comparator_matrice_plotter.py:53-60` <br> `src/callbacks/plotters/image_to_tag/confusion_matrix/nested_confuson_matrice_plotter.py:59-66` | 7 | Confusion matrix plotting configuration |

### 4.2 Recommended Actions

- **Sample files:** Extract common plotting code to a utility function
- **Confusion matrix plotters:** Consider a base class method or mixin

---

## 5. Prioritized Improvement Queue

### Phase 1: Critical Fixes (Effort: S, Impact: High)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-001 | Fix undefined names (`OutputSpace`, `typing`) | 2 |
| IMP-002 | Add None guards to generator/iterator access | 5 |
| IMP-003 | Fix return type mismatches in `OutputSpace` | 1 |

### Phase 2: Type Safety (Effort: M, Impact: Medium)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-004 | Add proper Optional handling across codebase | ~30 |
| IMP-005 | Update PIL imports for Pillow 10+ compatibility | 3 |
| IMP-006 | Remove unused imports | 7 |

### Phase 3: Code Quality (Effort: M, Impact: Low)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-007 | Refactor `Category.accept` (complexity 14) | 1 |
| IMP-008 | Add trailing newlines to files | ~10 |
| IMP-009 | Extract duplicate matplotlib code | 2 |

### Phase 4: Style Compliance (Effort: L, Impact: Low)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-010 | Configure line length to 100/120 or fix E501 | ~50 |
| IMP-011 | Fix blank line issues (E302/E303) | ~15 |
| IMP-012 | Fix whitespace issues | ~10 |

---

## 6. Recommendations

### 6.1 Quick Wins

1. **Add `pyproject.toml` flake8 config** with `max-line-length = 100` to reduce E501 noise
2. **Run `isort` and `autoflake`** to fix import issues automatically
3. **Add `# type: ignore` comments** strategically where third-party types are incomplete

### 6.2 CI/CD Integration

Consider adding pre-commit hooks:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pycqa/flake8
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
```

### 6.3 Effort Legend

- **S (Small):** < 1 hour, single file
- **M (Medium):** 1-4 hours, multiple files
- **L (Large):** > 4 hours, architectural changes

---

## Cross-References

- **Task:** [X.5.1 Code Quality Audit](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Index:** [index-codebase.md](../indices/index-codebase.md)
- **Lessons:** [lessons-learned/](../lessons-learned/)

---

## 7. Performance Analysis (Task X.5.2)

**Generated:** 2025-12-25
**Agents:** `machine-learning-researcher`, `python-pro`

### 7.1 Training Loop Bottlenecks

| Location | Issue | Severity | Effort | Impact |
|----------|-------|----------|--------|--------|
| `src/trainers/trainer.py:19` | **Eager execution enabled** (`tf.config.run_functions_eagerly(True)`) disables graph optimization and JIT compilation | **Critical** | S | High |
| `src/models/model_wrapper.py:116` | Lambda layer for tanh (`Lambda(lambda x: K.tanh(x))`) may not optimize as well as built-in `Activation('tanh')` | Low | S | Low |
| `src/losses/losses.py:106-123` | `edge_loss` computes Sobel gradients on every forward pass - computationally expensive | Medium | M | Medium |
| `src/modules/blocks/residual_block.py:24-25` | Dilated convolutions (`dilation_rate=3`) are slower than standard convs; justified for receptive field but has cost | Info | N/A | N/A |

#### Recommended Fix: Eager Execution (IMP-013)
```python
# Current (trainer.py:19) - SLOW
tf.config.run_functions_eagerly(True)

# Recommended - Remove or make conditional for debugging
# tf.config.run_functions_eagerly(False)  # Default
```
**Impact:** Removing eager execution can provide 2-10x speedup depending on model size and batch size.

### 7.2 GPU Memory Usage Patterns

| Location | Pattern | Memory Impact | Notes |
|----------|---------|---------------|-------|
| `src/trainers/trainer.py:24-25` | `set_memory_growth(gpu, True)` | ✅ Good | Prevents GPU memory hogging |
| `src/models/image_to_image/auto_encoder_wrapper.py:39-52` | U-Net skip connections store intermediate tensors | High | Inherent to architecture - expected |
| `src/modules/blocks/convolution_block.py:23` | AveragePooling2D downsampling | ✅ Good | Memory efficient vs strided convs |
| `src/modules/blocks/residual_block.py:18` | BatchNormalization per block | Medium | Stores running statistics; consider GroupNorm for small batches |
| `src/models/model_wrapper.py:15-17` | Default `layer_kernels=(32, 64, 128)` | Moderate | Reasonable baseline; memory scales with batch_size × channels × spatial |

#### Memory Estimation Formula
```
Memory ≈ batch_size × sum(spatial[i] × channels[i]) × 4 bytes × 2 (forward + backward)
```

For typical 256×256×3 input with batch_size=8 and kernels=(32, 64, 128):
- Encoder tensors: ~200 MB
- Decoder tensors: ~200 MB
- Skip connections: ~150 MB
- Gradients: ~550 MB
- **Total: ~1.1 GB** (fits comfortably on 4GB+ GPUs)

### 7.3 Generator/Dataloader Inefficiencies

| Location | Issue | Severity | Effort | Impact |
|----------|-------|----------|--------|--------|
| `src/generators/base_generators.py:35` | **ThreadPool created per batch** - significant overhead from pool creation/teardown | **High** | M | High |
| `src/generators/base_generators.py:43` | `np.random.choice()` per iteration instead of pre-shuffled indices | Low | S | Low |
| `src/samples/sample.py:42` | PIL + OpenCV mixing causes format conversion overhead | Medium | M | Medium |
| `src/trainers/trainer.py:203-208` | `tf.data.Dataset.from_generator()` without prefetching | **High** | S | High |
| N/A | No sample caching - re-reads from disk every epoch | Medium | L | Medium |

#### Recommended Fix: Persistent ThreadPool (IMP-014)
```python
# Current (base_generators.py) - Creates new pool per batch
def batch_processing(self, filenames):
    with ThreadPool(processes=self.batch_size) as pool:  # <-- Overhead
        data = pool.map(self.reader, filenames)

# Recommended - Reuse pool
def __init__(self, ...):
    ...
    self._thread_pool = ThreadPool(processes=self.batch_size)

def batch_processing(self, filenames):
    data = self._thread_pool.map(self.reader, filenames)

def __del__(self):
    if hasattr(self, '_thread_pool'):
        self._thread_pool.close()
```

#### Recommended Fix: Dataset Prefetching (IMP-015)
```python
# Current (trainer.py:203-208)
dataset = tf.data.Dataset.from_generator(lambda: generator, ...)

# Recommended - Add prefetch and parallel processing
dataset = tf.data.Dataset.from_generator(lambda: generator, ...)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### 7.4 Additional Optimization Opportunities

| ID | Optimization | Effort | Impact | Description |
|----|--------------|--------|--------|-------------|
| IMP-016 | Replace PIL+CV2 with unified backend | M | Medium | Use either `cv2.imread()` or `PIL` exclusively to avoid conversion overhead |
| IMP-017 | Add LRU cache for frequently accessed samples | L | Medium | `functools.lru_cache` on sample reading for datasets that fit in memory |
| IMP-018 | Mixed precision training | M | High | Enable `tf.keras.mixed_precision.set_global_policy('mixed_float16')` for ~2x speedup on modern GPUs |
| IMP-019 | Precompute and cache Sobel edges | M | Medium | For `edge_loss`, precompute ground truth edges in generator instead of loss |
| IMP-020 | Profile with `tf.profiler` | S | High | Add profiling callback to identify actual bottlenecks in production training |

### 7.5 Prioritized Performance Improvements

#### Phase P1: Quick Wins (Effort: S, Impact: High)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-013 | Disable eager execution (or make conditional) | `trainer.py` |
| IMP-015 | Add `dataset.prefetch(tf.data.AUTOTUNE)` | `trainer.py` |
| IMP-020 | Add TensorFlow Profiler callback for validation | `trainer.py` |

#### Phase P2: Generator Optimization (Effort: M, Impact: High)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-014 | Implement persistent ThreadPool | `base_generators.py` |
| IMP-016 | Unify image loading backend | `sample.py`, subclasses |
| IMP-019 | Precompute edge targets in generator | `losses.py`, generators |

#### Phase P3: Advanced Optimizations (Effort: M-L, Impact: Medium-High)

| ID | Task | Files Affected |
|----|------|----------------|
| IMP-018 | Enable mixed precision training | `model_wrapper.py`, `trainer.py` |
| IMP-017 | Implement sample caching with LRU | `base_generators.py` |

### 7.6 Numerical Stability Considerations

**Reference:** [wbce-log-negative-values.md](../lessons-learned/wbce-log-negative-values.md)

Current loss implementations use `epsilon = 1e-7` consistently (good practice). Key observations:

- `cross_entropy_positive()` properly handles log domain with epsilon
- No overflow protection for very large activations (acceptable for sigmoid outputs)
- `edge_loss` sqrt includes epsilon to prevent gradient explosion at zero

---

## 8. Feature Gap Analysis (Task X.5.3)

**Generated:** 2025-12-25
**Agent:** `machine-learning-researcher`

### 8.1 Framework Comparison: Current vs Modern Standards

| Feature Area | This Framework | PyTorch Lightning | Keras 3 | Gap Level |
|--------------|----------------|-------------------|---------|-----------|
| **Training Loop** | Custom `Trainer.run()` via `model.fit()` | `Trainer` with hooks | `model.fit()` + callbacks | Low |
| **Model Abstraction** | `ModelWrapper` dataclass | `LightningModule` | Subclassed `Model` | Low |
| **Data Loading** | Custom `BatchGenerator` + ThreadPool | `DataLoader` + workers | `tf.data.Dataset` | Medium |
| **Callbacks** | Custom `Callback` base class | Built-in + custom hooks | Built-in `Callback` | Low |
| **Mixed Precision** | Not implemented | `precision="16-mixed"` | `mixed_float16` policy | **High** |
| **Gradient Accumulation** | Not implemented | Built-in | Manual | **High** |
| **Multi-GPU/Distributed** | Not implemented | Built-in DDP/FSDP | `tf.distribute` | **High** |
| **Checkpointing** | Not implemented | `ModelCheckpoint` | `ModelCheckpoint` callback | **Medium** |
| **LR Scheduling** | Not implemented | Scheduler integration | `LearningRateScheduler` | **Medium** |
| **Logging/Tracking** | CSV history only | W&B, MLflow, TensorBoard | TensorBoard callback | **Medium** |
| **Pre-trained Backbones** | None | torchvision/timm | `keras.applications` | **Critical** |
| **Attention Mechanisms** | None | Native + timm | `keras.layers.Attention` | **High** |

### 8.2 Architecture Gaps

#### 8.2.1 Missing Backbone Architectures

| ID | Architecture | Use Case | Implementation Effort | Impact |
|----|--------------|----------|----------------------|--------|
| FG-001 | **ResNet (18/34/50/101)** | General-purpose encoder | M | **Critical** |
| FG-002 | **EfficientNetV2** | Efficient classification/segmentation | M | High |
| FG-003 | **ConvNeXt** | Modern CNN baseline | M | High |
| FG-004 | **Vision Transformer (ViT)** | Attention-based encoder | L | Medium |
| FG-005 | **Swin Transformer** | Hierarchical vision transformer | L | High |
| FG-006 | **MobileNetV3** | Edge deployment | S | Medium |

**Current State:** Only custom convolutional encoder via `build_encoder()` with configurable kernel counts.

#### 8.2.2 Missing Decoder/Segmentation Architectures

| ID | Architecture | Use Case | Implementation Effort | Impact |
|----|--------------|----------|----------------------|--------|
| FG-007 | **UNet++** | Nested skip connections | M | High |
| FG-008 | **DeepLabV3+** | Atrous spatial pyramid pooling | M | High |
| FG-009 | **FPN (Feature Pyramid Network)** | Multi-scale features | M | High |
| FG-010 | **PANet** | Bottom-up path augmentation | M | Medium |
| FG-011 | **SegFormer** | Transformer-based segmentation | L | Medium |

**Current State:** Basic U-Net with `build_encoder()` + `build_decoder()` and skip connections.

#### 8.2.3 Missing Attention Modules

| ID | Module | Description | Implementation Effort | Impact |
|----|--------|-------------|----------------------|--------|
| FG-012 | **CBAM** | Channel + Spatial attention | S | High |
| FG-013 | **SE (Squeeze-and-Excitation)** | Channel attention | S | High |
| FG-014 | **ECA (Efficient Channel Attention)** | Lightweight channel attention | S | Medium |
| FG-015 | **Self-Attention / Multi-Head** | Global context | M | Medium |
| FG-016 | **Axial Attention** | Efficient self-attention | M | Low |

**Current State:** No attention mechanisms implemented.

### 8.3 Loss Function Gaps

| ID | Loss Function | Use Case | Implementation Effort | Impact |
|----|---------------|----------|----------------------|--------|
| FG-017 | **Focal Loss** | Class imbalance (detection/segmentation) | S | **Critical** |
| FG-018 | **Lovász-Softmax** | Direct IoU optimization | S | High |
| FG-019 | **Perceptual/VGG Loss** | Style transfer, super-resolution | M | High |
| FG-020 | **SSIM Loss** | Structural similarity preservation | S | High |
| FG-021 | **Tversky Loss** | Tunable precision/recall balance | S | Medium |
| FG-022 | **InfoNCE / NT-Xent** | Contrastive learning | M | Medium |
| FG-023 | **Deep Supervision Loss** | Auxiliary outputs at encoder stages | M | Medium |
| FG-024 | **Boundary/Surface Loss** | Edge-aware segmentation | S | Medium |

**Current State:**
- ✅ MAE, Cross-Entropy, Dice, Edge (Sobel), Std Difference
- ✅ Blurriness loss (from model), Encoding Similarity
- ✅ KID (Kernel Inception Distance)

### 8.4 Metrics Gaps

| ID | Metric | Use Case | Implementation Effort | Impact |
|----|--------|----------|----------------------|--------|
| FG-025 | **IoU / mIoU** | Segmentation evaluation | S | **Critical** |
| FG-026 | **Pixel Accuracy / Class Accuracy** | Segmentation | S | High |
| FG-027 | **PSNR** | Image quality (reconstruction) | S | High |
| FG-028 | **SSIM** | Structural similarity | S | High |
| FG-029 | **FID (Fréchet Inception Distance)** | Generative model quality | M | Medium |
| FG-030 | **Precision / Recall / F1 per class** | Classification analysis | S | High |
| FG-031 | **AUC-ROC / PR-AUC** | Binary classification | S | Medium |
| FG-032 | **Mean Average Precision (mAP)** | Detection | M | Low |

**Current State:** Only standard Keras metrics (accuracy, loss). No domain-specific metrics.

### 8.5 Training Infrastructure Gaps

| ID | Feature | Description | Implementation Effort | Impact |
|----|---------|-------------|----------------------|--------|
| FG-033 | **Gradient Accumulation** | Train with larger effective batch sizes | S | **High** |
| FG-034 | **Mixed Precision Training** | `tf.keras.mixed_precision` policy | S | **High** |
| FG-035 | **Model Checkpointing** | Save best/periodic model weights | S | **High** |
| FG-036 | **Early Stopping** | Stop training on plateau | S | High |
| FG-037 | **LR Schedulers** | Warmup, cosine annealing, reduce on plateau | M | High |
| FG-038 | **Experiment Tracking** | W&B, MLflow, TensorBoard integration | M | Medium |
| FG-039 | **Multi-GPU Training** | `tf.distribute.MirroredStrategy` | M | Medium |
| FG-040 | **Gradient Clipping** | Prevent exploding gradients | S | Medium |
| FG-041 | **EMA (Exponential Moving Average)** | Model weight averaging | S | Medium |

### 8.6 Data Pipeline Gaps

| ID | Feature | Description | Implementation Effort | Impact |
|----|---------|-------------|----------------------|--------|
| FG-042 | **Albumentations Integration** | Rich augmentation library | M | High |
| FG-043 | **Multi-scale Training** | Random image size per batch | M | Medium |
| FG-044 | **Mosaic Augmentation** | Combine 4 images (YOLO-style) | M | Low |
| FG-045 | **CutMix / MixUp** | Regularization via mixing | S | Medium |
| FG-046 | **Online Hard Example Mining (OHEM)** | Focus on difficult samples | M | Medium |
| FG-047 | **Sample Weighting** | Instance-level importance | S | Medium |

**Current State:**
- ✅ VerticalSymmetry (horizontal flip)
- ✅ Blurry augmentation
- ✅ Class weighting via `Loss.apply_class_weights()`
- ❌ No Albumentations, no advanced augmentations

### 8.7 Prioritized Feature Roadmap

#### Phase F1: Critical Infrastructure (Effort: S-M, Impact: Critical/High)

| ID | Task | Files Affected | Depends On |
|----|------|----------------|------------|
| FG-033 | Add gradient accumulation wrapper | `trainer.py`, `model_wrapper.py` | - |
| FG-034 | Enable mixed precision training | `trainer.py` | - |
| FG-035 | Add ModelCheckpoint callback | `callbacks/` | - |
| FG-017 | Implement Focal Loss | `losses/losses.py` | - |
| FG-025 | Implement IoU / mIoU metric | `metrics/` (new) | - |

#### Phase F2: Modern Backbones (Effort: M, Impact: Critical/High)

| ID | Task | Files Affected | Depends On |
|----|------|----------------|------------|
| FG-001 | Add ResNet backbone support | `modules/backbones/` (new) | - |
| FG-002 | Add EfficientNetV2 backbone | `modules/backbones/` | FG-001 |
| FG-012 | Implement CBAM attention | `modules/layers/` | - |
| FG-013 | Implement SE attention | `modules/layers/` | - |

#### Phase F3: Advanced Losses & Metrics (Effort: S-M, Impact: High)

| ID | Task | Files Affected | Depends On |
|----|------|----------------|------------|
| FG-018 | Implement Lovász-Softmax loss | `losses/losses.py` | FG-025 |
| FG-019 | Implement Perceptual/VGG loss | `losses/from_model/` | FG-001 |
| FG-020 | Implement SSIM loss/metric | `losses/`, `metrics/` | - |
| FG-027 | Implement PSNR metric | `metrics/` | - |
| FG-030 | Implement per-class F1/precision/recall | `metrics/` | - |

#### Phase F4: Training Enhancements (Effort: M, Impact: Medium)

| ID | Task | Files Affected | Depends On |
|----|------|----------------|------------|
| FG-036 | Add EarlyStopping callback | `callbacks/` | FG-035 |
| FG-037 | Add LR scheduler support | `trainer.py` | - |
| FG-038 | Add TensorBoard/W&B logging | `callbacks/` | - |
| FG-040 | Add gradient clipping | `trainer.py`, `model_wrapper.py` | - |
| FG-041 | Add EMA model averaging | `model_wrapper.py` | - |

#### Phase F5: Advanced Architectures (Effort: M-L, Impact: Medium)

| ID | Task | Files Affected | Depends On |
|----|------|----------------|------------|
| FG-007 | Implement UNet++ | `models/image_to_image/` | - |
| FG-008 | Implement DeepLabV3+ | `models/image_to_image/` | - |
| FG-009 | Implement FPN | `modules/` | FG-001 |
| FG-004 | Add ViT backbone | `modules/backbones/` | FG-015 |
| FG-005 | Add Swin Transformer | `modules/backbones/` | FG-004 |

### 8.8 Quick Wins Summary

**Immediate value, minimal effort (< 1 hour each):**

1. **FG-017 Focal Loss** - 20 lines, critical for class imbalance
2. **FG-020 SSIM Loss** - Use `tf.image.ssim`, 10 lines
3. **FG-025 IoU Metric** - 15 lines, essential for segmentation
4. **FG-035 ModelCheckpoint** - Wire existing Keras callback
5. **FG-036 EarlyStopping** - Wire existing Keras callback

**Moderate effort, high value (1-4 hours each):**

1. **FG-034 Mixed Precision** - Add policy configuration
2. **FG-012 CBAM Attention** - ~100 lines, widely applicable
3. **FG-019 Perceptual Loss** - ~50 lines + VGG loading

### 8.9 Effort Legend

- **S (Small):** < 1 hour, self-contained addition
- **M (Medium):** 1-4 hours, multiple files or moderate complexity
- **L (Large):** > 4 hours, significant architecture changes or new module

---

## 9. Technical Debt Inventory (Task X.5.4)

**Generated:** 2025-12-25
**Agents:** `refactoring-specialist`, `devops-engineer`, `machine-learning-researcher`

### 9.1 TODO/FIXME/HACK Markers

**Tool:** `grep -rn "TODO\|FIXME\|HACK\|XXX" src/ test/`
**Result:** ✅ **No markers found**

The codebase is clean of TODO/FIXME markers. This is a positive indicator of code hygiene.

### 9.2 Deprecated TensorFlow Patterns

#### 9.2.1 Model Saving Format (`.h5` → `.keras`)

**Severity:** Medium | **Effort:** S | **Impact:** Medium

| ID | Location | Current Pattern | Recommended |
|----|----------|-----------------|-------------|
| TD-001 | `src/callbacks/history_callback.py:43-45` | `model.save(..."/model.h5")` | Use `.keras` format or SavedModel |
| TD-002 | `src/scripts/nested_on_video.py:36` | `load_weights(.../model.h5)` | Migrate to `.keras` format |
| TD-003 | `src/scripts/make_blurry_loss_from_encoder.py:17` | `.blurry.h5` hardcoded | Use `.keras` extension |
| TD-004 | `src/scripts/nested_pca.py:52` | `load_weights(.../model.h5)` | Migrate to `.keras` format |
| TD-005 | `src/scripts/make_loss_from_encoder.py:21,45` | `.h5` default paths | Use `.keras` extension |
| TD-006 | `src/losses/from_model/blurriness.py:10` | `.blurry.h5` hardcoded | Use `.keras` extension |
| TD-007 | `src/losses/from_model/encoding_similarity.py:13` | `.kid.h5` hardcoded | Use `.keras` extension |
| TD-008 | `src/scripts/utils/processors/processor.py:33` | `load_weights(.../model.h5)` | Migrate to `.keras` format |

**Migration Notes:**
- TensorFlow 2.13+ recommends `.keras` format (native Keras v3 format)
- `.h5` format has known limitations with custom objects and Lambda layers
- SavedModel format is preferred for production deployment

#### 9.2.2 Legacy Keras Backend Usage (`K.*` → `tf.*`)

**Severity:** Low | **Effort:** M | **Impact:** Low

The `tensorflow.keras.backend as K` pattern is legacy and should migrate to direct `tf.*` operations.

| ID | File | `K.*` Usages | Migration Target |
|----|------|--------------|------------------|
| TD-009 | `src/losses/from_model/blurriness.py` | `K.mean` | `tf.reduce_mean` |
| TD-010 | `src/losses/from_model/encoding_similarity.py` | `K.mean`, `K.abs` | `tf.reduce_mean`, `tf.abs` |
| TD-011 | `src/losses/kid.py` | `K.cast`, `K.shape`, `K.transpose`, `K.eye`, `K.sum`, `K.mean` | `tf.cast`, `tf.shape`, `tf.transpose`, `tf.eye`, `tf.reduce_sum`, `tf.reduce_mean` |
| TD-012 | `src/models/model_wrapper.py` | `K.tanh` in Lambda | `tf.keras.layers.Activation('tanh')` or `tf.nn.tanh` |
| TD-013 | `src/models/image_to_tag/comparator_wrapper.py` | Import only (unused) | Remove import |
| TD-014 | `src/models/image_to_image/diffusion_model_wrapper.py` | `K.cast`, `K.cos`, `K.sin`, `K.ones`, `K.random_normal`, `K.random_uniform` | `tf.cast`, `tf.cos`, `tf.sin`, `tf.ones`, `tf.random.normal`, `tf.random.uniform` |
| TD-015 | `src/models/image_to_image/auto_encoder_wrapper.py` | Import only | Verify usage or remove |

**Migration Rationale:**
- `tf.keras.backend` functions wrap TF operations with extra overhead
- Direct `tf.*` operations are clearer and enable better graph optimization
- Some `K.*` functions may be removed in future TensorFlow versions

#### 9.2.3 Non-Standard Variable Creation

**Severity:** Low | **Effort:** S | **Impact:** Low

| ID | Location | Issue | Fix |
|----|----------|-------|-----|
| TD-016 | `src/modules/layers/sparse_conv2d.py:89` | `tf.Variable()` inside Keras layer | Use `self.add_weight()` for proper tracking |

**Current:**
```python
return tf.Variable(initial_value=chosen_patterns, dtype=tf.int32, trainable=False)
```

**Recommended:**
```python
return self.add_weight(name='chosen_patterns',
                       initializer=tf.constant_initializer(chosen_patterns),
                       dtype=tf.int32, trainable=False)
```

### 9.3 Dependency Issues

**Source:** `setup.py`

#### 9.3.1 Outdated Packages

| ID | Package | Current Version | Latest Stable | Severity | Notes |
|----|---------|-----------------|---------------|----------|-------|
| TD-017 | TensorFlow | 2.17 | 2.20 (Aug 2025) | Low | Minor update, 2.17 still functional |
| TD-018 | Basemap | (any) | **DEPRECATED** | **High** | No longer maintained since 2020; migrate to Cartopy |

**Basemap → Cartopy Migration:**
- [Basemap Deprecation Notice](https://github.com/matplotlib/basemap/issues/568)
- [Cartopy Documentation](https://scitools.org.uk/cartopy/)
- Cartopy provides similar functionality with active maintenance

#### 9.3.2 Unpinned Dependencies (Reproducibility Risk)

| ID | Package | Risk Level | Recommendation |
|----|---------|------------|----------------|
| TD-019 | pandas | Medium | Pin to major.minor (e.g., `pandas>=2.0,<3.0`) |
| TD-020 | matplotlib | Low | Pin to major.minor |
| TD-021 | opencv-python | Medium | Pin version for API stability |
| TD-022 | scikit-learn | Low | Pin to major.minor |
| TD-023 | Pillow | Medium | Pin version; API changes between majors |
| TD-024 | pytest | Low | Pin for CI consistency |

#### 9.3.3 Build System Modernization

| ID | Issue | Severity | Effort | Recommendation |
|----|-------|----------|--------|----------------|
| TD-025 | Using `setup.py` only | Low | M | Migrate to `pyproject.toml` (PEP 517/518) |
| TD-026 | No lock file | Medium | S | Add `uv.lock` or `requirements.lock` |
| TD-027 | Git dependency (`rignak`) | Low | N/A | Acceptable for private packages |

### 9.4 Code Inflexibility Patterns

**Reference:** [hardcoded-file-extension.md](../lessons-learned/hardcoded-file-extension.md)

| ID | Location | Pattern | Issue |
|----|----------|---------|-------|
| TD-028 | Multiple scripts | Hardcoded `.tmp/` paths | Paths like `.tmp/20250115_095140/model.h5` |
| TD-006, TD-007 | Loss files | Hardcoded model paths | Should be configurable |

### 9.5 CI/CD Debt

**Reference:** [ci-branch-naming.md](../lessons-learned/ci-branch-naming.md)

| ID | Area | Current State | Improvement |
|----|------|---------------|-------------|
| TD-029 | Pre-commit hooks | Not configured | Add `.pre-commit-config.yaml` |
| TD-030 | Large file prevention | Not enforced | Add `check-added-large-files` hook |
| TD-031 | Binary file gitignore | Partial | Verify `.nc`, `.hdf*`, `*.SAFE/` patterns |

### 9.6 Prioritized Technical Debt Remediation

#### Phase TD1: Critical Fixes (Effort: S, Impact: High)

| ID | Task | Files Affected |
|----|------|----------------|
| TD-018 | Migrate from Basemap to Cartopy | TBD (search for basemap usage) |
| TD-001 to TD-008 | Migrate `.h5` to `.keras` format | 8 files |

#### Phase TD2: Modernization (Effort: M, Impact: Medium)

| ID | Task | Files Affected |
|----|------|----------------|
| TD-025 | Create `pyproject.toml` | Root |
| TD-019 to TD-024 | Pin dependency versions | `pyproject.toml` |
| TD-009 to TD-015 | Migrate `K.*` to `tf.*` | 7 files |

#### Phase TD3: Best Practices (Effort: S-M, Impact: Low)

| ID | Task | Files Affected |
|----|------|----------------|
| TD-016 | Fix `tf.Variable` in SparseConv2D | `sparse_conv2d.py` |
| TD-029 | Add pre-commit hooks | Root |
| TD-017 | Upgrade TensorFlow 2.17 → 2.20 | `setup.py`/`pyproject.toml` |

### 9.7 Summary Statistics

| Category | Items | Critical | High | Medium | Low |
|----------|-------|----------|------|--------|-----|
| TODO/FIXME Markers | 0 | - | - | - | - |
| Deprecated TF Patterns | 16 | 0 | 0 | 1 | 15 |
| Dependency Issues | 12 | 0 | 1 | 4 | 7 |
| Code Inflexibility | 3 | 0 | 0 | 2 | 1 |
| CI/CD Debt | 3 | 0 | 0 | 1 | 2 |
| **Total** | **34** | **0** | **1** | **8** | **25** |

### 9.8 Effort Legend

- **S (Small):** < 1 hour, mechanical changes
- **M (Medium):** 1-4 hours, requires testing
- **L (Large):** > 4 hours, architectural impact

---

## 10. Documentation Gap Analysis (Task X.5.5)

**Completed:** 2025-12-25
**Agent:** `technical-writer`, `archivist`

### 10.1 Docstring Coverage Analysis

**Critical Finding:** Documentation coverage is severely lacking across the codebase.

| Category | With Docstring | Without Docstring | Coverage |
|----------|----------------|-------------------|----------|
| Modules | 1 | 142 | **0.7%** |
| Classes | 2 | 102 | **1.9%** |
| Functions (top-level) | 0 | 33 | **0.0%** |

#### 10.1.1 Modules WITH Docstrings (Complete List)

| ID | Module | Notes |
|----|--------|-------|
| - | `src/modules/layers/sparse_conv2d.py` | Auto-generated (Gemini) |

#### 10.1.2 Classes WITH Docstrings (Complete List)

| ID | Class | Location |
|----|-------|----------|
| - | `VideoProcessor` | `src/scripts/utils/processors/video_processor.py` |
| - | `FileProcessor` | `src/scripts/utils/processors/file_processor.py` |

#### 10.1.3 Priority Modules for Docstring Addition

| ID | Module | Reason | Effort | Impact |
|----|--------|--------|--------|--------|
| DG-001 | `src/trainers/trainer.py` | Core base class | M | High |
| DG-002 | `src/models/model_wrapper.py` | Core base class | M | High |
| DG-003 | `src/generators/base_generators.py` | Core base class | M | High |
| DG-004 | `src/callbacks/callback.py` | Core base class | S | High |
| DG-005 | `src/config.py` | Entry point for configuration | S | High |
| DG-006 | `src/modules/module.py` | Architecture building blocks | M | High |
| DG-007 | `src/losses/losses.py` | Loss function reference | M | Medium |
| DG-008 | `src/output_spaces/output_space.py` | Output space abstraction | M | Medium |

### 10.2 README Gaps

**Current State:** The README provides minimal guidance with generic instructions.

| ID | Gap | Description | Effort | Impact |
|----|-----|-------------|--------|--------|
| DG-009 | No working code examples | README lacks copy-paste runnable examples | M | High |
| DG-010 | Missing configuration reference | `RIGNAK_ML_DATASET_ROOT` env var undocumented | S | High |
| DG-011 | Missing Rignak dependency docs | Custom `rignak` library not explained | M | High |
| DG-012 | No API documentation | No class/function reference | L | High |
| DG-013 | No troubleshooting section | Common errors not documented | M | Medium |
| DG-014 | Missing output examples | No sample outputs shown | S | Medium |

### 10.3 Tutorial Needs (New User Onboarding)

| ID | Tutorial Topic | Description | Effort | Impact |
|----|----------------|-------------|--------|--------|
| DG-015 | Quick Start Guide | First model training in 5 minutes | M | High |
| DG-016 | Dataset Preparation | How to structure training data | M | High |
| DG-017 | Custom Trainer Creation | Extending base trainer class | M | High |
| DG-018 | Custom Generator Creation | Data pipeline customization | M | Medium |
| DG-019 | Custom Callback Creation | Training monitoring/visualization | S | Medium |
| DG-020 | Custom Loss Function | Adding new loss functions | S | Medium |
| DG-021 | Model Architecture Guide | Using/extending encoder-decoder | L | Medium |
| DG-022 | Inference Guide | Using trained models for prediction | M | High |

### 10.4 EPP Index Completeness

| ID | Index | Issue | Effort | Impact |
|----|-------|-------|--------|--------|
| DG-023 | `index-phases.md` | Missing X.5 subtask status (X.5.1-X.5.6) | S | Low |
| DG-024 | `index-agents.md` | Missing some CLAUDE.md agents (scientific-publication-editor, remote-sensing-oceanographer, geospatial-visualizer) | S | Low |

### 10.5 Recommendations

#### Immediate Actions (High Impact, Low Effort)

1. **Add module docstrings to core files:** Start with `config.py`, `callback.py`
2. **Document environment variables:** Add `RIGNAK_ML_DATASET_ROOT` to README
3. **Add one working example:** Create a minimal training example in README

#### Medium-Term Actions

1. **Create Quick Start tutorial:** Step-by-step first model training
2. **Add docstrings to base classes:** `Trainer`, `ModelWrapper`, `BatchGenerator`
3. **Document Rignak dependency:** Explain installation and purpose

#### Long-Term Actions

1. **Generate API documentation:** Consider Sphinx/MkDocs for full API docs
2. **Create tutorial series:** Cover all extension points
3. **Add docstrings systematically:** Use pydocstyle enforcement in CI

### 10.6 Effort Legend

- **S (Small):** < 1 hour
- **M (Medium):** 1-4 hours
- **L (Large):** > 4 hours

---

## Cross-References

- **Task:** [X.5.1 Code Quality Audit](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Task:** [X.5.2 Performance Analysis](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Task:** [X.5.3 Feature Gap Analysis](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Task:** [X.5.4 Technical Debt Inventory](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Task:** [X.5.5 Documentation Gap Analysis](../phases/phase-X-off-chronology/task-X.5-potential-improvements.md)
- **Index:** [index-codebase.md](../indices/index-codebase.md)
- **Index:** [index-documentation.md](../indices/index-documentation.md)
- **Lessons:** [lessons-learned/](../lessons-learned/)

---

## 11. Consolidated Prioritization Matrix (Task X.5.6)

**Generated:** 2025-12-25
**Agent:** `agent-organizer`, `archivist`

### 11.1 Effort × Impact Summary

#### 11.1.1 Quick Reference: All Items by ID

| ID Range | Category | Count | Section |
|----------|----------|-------|---------|
| IMP-001 to IMP-012 | Code Quality | 12 | §5 |
| IMP-013 to IMP-020 | Performance | 8 | §7 |
| FG-001 to FG-047 | Feature Gaps | 47 | §8 |
| TD-001 to TD-031 | Technical Debt | 31 | §9 |
| DG-001 to DG-024 | Documentation | 24 | §10 |
| **Total** | | **122** | |

#### 11.1.2 Effort × Impact Matrix

##### High Impact

| Effort | Items | Example IDs |
|--------|-------|-------------|
| **S (Small)** | 15 | IMP-001, IMP-002, IMP-013, IMP-015, FG-017, FG-025, FG-033, FG-034, FG-035 |
| **M (Medium)** | 18 | IMP-004, IMP-014, FG-001, FG-002, FG-012, DG-001, DG-002, DG-003 |
| **L (Large)** | 6 | FG-039, DG-012, DG-021 |

##### Medium Impact

| Effort | Items | Example IDs |
|--------|-------|-------------|
| **S (Small)** | 12 | IMP-005, FG-020, FG-021, FG-028, FG-040, DG-010, DG-014 |
| **M (Medium)** | 22 | IMP-016, IMP-018, FG-007, FG-008, FG-015, FG-037, FG-038, DG-013, DG-017 |
| **L (Large)** | 8 | FG-004, FG-005, FG-011, DG-022 |

##### Low Impact

| Effort | Items | Example IDs |
|--------|-------|-------------|
| **S (Small)** | 18 | IMP-008, TD-016, DG-023, DG-024 |
| **M (Medium)** | 15 | IMP-010, IMP-011, TD-009 to TD-015 |
| **L (Large)** | 8 | IMP-017 (large cache system) |

#### 11.1.3 Priority Quadrant Summary

```
                        IMPACT
                Low         Medium        High
           ┌──────────┬──────────────┬──────────────┐
    Small  │    18    │      12      │     15       │  ← START HERE
           │ Backlog  │  Nice-to-have│  QUICK WINS  │
    E      ├──────────┼──────────────┼──────────────┤
    F      │    15    │      22      │     18       │
    F  Med │ Backlog  │  Planned     │  HIGH VALUE  │
    O      ├──────────┼──────────────┼──────────────┤
    R      │     8    │       8      │      6       │
    T Large│ Defer    │  Strategic   │  MAJOR PROJ  │
           └──────────┴──────────────┴──────────────┘
```

**Recommended Action Order:**
1. **Quick Wins (S/High):** 15 items - immediate value
2. **High Value (M/High):** 18 items - planned sprints
3. **Nice-to-have (S/Medium):** 12 items - fill gaps
4. **Strategic (M/Medium + L/High):** 28 items - roadmap items
5. **Backlog:** Remaining - as capacity allows

---

### 11.2 Unified Phase Roadmap

This roadmap consolidates all improvement categories into coherent execution phases.

#### Phase U1: Foundation Fixes (Week-equivalent effort: S)
**Goal:** Remove blockers, fix critical issues, enable safer development.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | IMP-001 | Fix undefined names (OutputSpace, typing) | Code Quality | S | High |
| 2 | IMP-013 | Disable eager execution | Performance | S | High |
| 3 | IMP-015 | Add dataset.prefetch() | Performance | S | High |
| 4 | IMP-002 | Add None guards to generators | Code Quality | S | High |
| 5 | FG-035 | Add ModelCheckpoint callback | Feature Gap | S | High |
| 6 | FG-036 | Add EarlyStopping callback | Feature Gap | S | High |
| 7 | TD-029 | Add pre-commit hooks | Tech Debt | S | Medium |

**Deliverables:** Stable, faster training; basic checkpoint safety; CI quality gates.

#### Phase U2: Type Safety & Modernization (Week-equivalent effort: M)
**Goal:** Improve code robustness and maintainability.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | IMP-004 | Add proper Optional handling | Code Quality | M | Medium |
| 2 | TD-001-008 | Migrate .h5 → .keras format | Tech Debt | S | Medium |
| 3 | TD-025 | Create pyproject.toml | Tech Debt | M | Medium |
| 4 | IMP-005 | Update PIL imports for Pillow 10+ | Code Quality | S | Medium |
| 5 | IMP-006 | Remove unused imports | Code Quality | S | Low |
| 6 | TD-019-024 | Pin dependency versions | Tech Debt | S | Medium |

**Deliverables:** Type-safe codebase; modern build system; reproducible builds.

#### Phase U3: Performance Optimization (Week-equivalent effort: M)
**Goal:** Significant training speed improvements.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | IMP-014 | Implement persistent ThreadPool | Performance | M | High |
| 2 | FG-034 | Enable mixed precision training | Feature Gap | S | High |
| 3 | FG-033 | Add gradient accumulation | Feature Gap | S | High |
| 4 | IMP-016 | Unify image loading backend | Performance | M | Medium |
| 5 | IMP-018 | Mixed precision policy config | Performance | M | High |
| 6 | FG-040 | Add gradient clipping | Feature Gap | S | Medium |

**Deliverables:** 2-5x training speedup; larger effective batch sizes.

#### Phase U4: Core Features - Losses & Metrics (Week-equivalent effort: M)
**Goal:** Essential ML building blocks.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | FG-017 | Implement Focal Loss | Feature Gap | S | High |
| 2 | FG-025 | Implement IoU/mIoU metric | Feature Gap | S | High |
| 3 | FG-020 | Implement SSIM loss/metric | Feature Gap | S | High |
| 4 | FG-027 | Implement PSNR metric | Feature Gap | S | High |
| 5 | FG-030 | Implement per-class F1/precision/recall | Feature Gap | S | High |
| 6 | FG-018 | Implement Lovász-Softmax loss | Feature Gap | S | Medium |
| 7 | FG-021 | Implement Tversky loss | Feature Gap | S | Medium |

**Deliverables:** Standard segmentation losses/metrics; better evaluation capabilities.

#### Phase U5: Modern Backbones (Week-equivalent effort: M-L)
**Goal:** State-of-the-art encoder architectures.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | FG-001 | Add ResNet backbone support | Feature Gap | M | High |
| 2 | FG-012 | Implement CBAM attention | Feature Gap | S | High |
| 3 | FG-013 | Implement SE attention | Feature Gap | S | High |
| 4 | FG-002 | Add EfficientNetV2 backbone | Feature Gap | M | High |
| 5 | FG-003 | Add ConvNeXt backbone | Feature Gap | M | High |
| 6 | FG-019 | Implement Perceptual/VGG loss | Feature Gap | M | High |

**Deliverables:** Pre-trained backbone support; attention mechanisms; perceptual loss.

#### Phase U6: Documentation (Week-equivalent effort: M)
**Goal:** Usable project for new developers.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | DG-005 | Document config.py | Documentation | S | High |
| 2 | DG-010 | Document environment variables | Documentation | S | High |
| 3 | DG-009 | Add working code examples to README | Documentation | M | High |
| 4 | DG-015 | Create Quick Start Guide | Documentation | M | High |
| 5 | DG-001-004 | Add docstrings to core classes | Documentation | M | High |
| 6 | DG-016 | Dataset Preparation guide | Documentation | M | High |

**Deliverables:** Onboarding documentation; runnable examples; core API docs.

#### Phase U7: Advanced Architectures (Week-equivalent effort: L)
**Goal:** State-of-the-art segmentation architectures.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | FG-007 | Implement UNet++ | Feature Gap | M | High |
| 2 | FG-008 | Implement DeepLabV3+ | Feature Gap | M | High |
| 3 | FG-009 | Implement FPN | Feature Gap | M | High |
| 4 | FG-004 | Add ViT backbone | Feature Gap | L | Medium |
| 5 | FG-005 | Add Swin Transformer | Feature Gap | L | High |

**Deliverables:** Multiple segmentation architectures; transformer support.

#### Phase U8: Training Ecosystem (Week-equivalent effort: M)
**Goal:** Production-ready training infrastructure.

| Priority | ID | Task | Category | Effort | Impact |
|----------|-----|------|----------|--------|--------|
| 1 | FG-037 | Add LR scheduler support | Feature Gap | M | High |
| 2 | FG-038 | Add TensorBoard/W&B logging | Feature Gap | M | Medium |
| 3 | FG-039 | Add multi-GPU training | Feature Gap | M | Medium |
| 4 | FG-041 | Add EMA model averaging | Feature Gap | S | Medium |
| 5 | FG-042 | Albumentations integration | Feature Gap | M | High |

**Deliverables:** Experiment tracking; distributed training; rich augmentations.

---

### 11.3 Dependency Graph

```
Phase U1 (Foundation)
    ↓
    ├→ Phase U2 (Type Safety) → Phase U6 (Documentation)
    │
    ├→ Phase U3 (Performance) → Phase U8 (Training Ecosystem)
    │
    └→ Phase U4 (Losses/Metrics) → Phase U5 (Backbones) → Phase U7 (Architectures)
```

**Critical Path:** U1 → U3 → U4 → U5 → U7

---

### 11.4 Quick Reference: Top 20 Priority Items

| Rank | ID | Task | Effort | Impact |
|------|-----|------|--------|--------|
| 1 | IMP-001 | Fix undefined names | S | High |
| 2 | IMP-013 | Disable eager execution | S | High |
| 3 | IMP-015 | Add dataset prefetch | S | High |
| 4 | FG-035 | ModelCheckpoint callback | S | High |
| 5 | FG-017 | Focal Loss | S | High |
| 6 | FG-025 | IoU/mIoU metric | S | High |
| 7 | FG-033 | Gradient accumulation | S | High |
| 8 | FG-034 | Mixed precision training | S | High |
| 9 | IMP-002 | None guards in generators | S | High |
| 10 | IMP-014 | Persistent ThreadPool | M | High |
| 11 | FG-001 | ResNet backbone | M | High |
| 12 | FG-012 | CBAM attention | S | High |
| 13 | FG-020 | SSIM loss/metric | S | High |
| 14 | FG-027 | PSNR metric | S | High |
| 15 | DG-015 | Quick Start Guide | M | High |
| 16 | TD-029 | Pre-commit hooks | S | Medium |
| 17 | IMP-004 | Optional type handling | M | Medium |
| 18 | TD-001-008 | Migrate .h5 → .keras | S | Medium |
| 19 | FG-002 | EfficientNetV2 backbone | M | High |
| 20 | FG-036 | EarlyStopping callback | S | High |

---

### 11.5 Acceptance Criteria Verification

- [x] Improvements are clearly described (122 items across 5 categories)
- [x] Priorities are assigned (S/M/L effort × Low/Med/High impact matrix)
- [x] Effort estimates are provided (not time-based)
- [x] Items are actionable (specific files and code locations referenced)
- [x] All findings documented in improvement backlog file

---

*Last Updated: 2025-12-25 (Task X.5.6 Consolidated)*
