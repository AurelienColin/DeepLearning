# ML Framework - Project Overview

> Comprehensive architecture and design documentation for the Image-to-Image and Image-to-Tag Deep Learning Framework.

**Status**: In Progress
**Last Updated**: 2025-12-25

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Flow Patterns](#2-data-flow-patterns)
3. [Extension Points](#3-extension-points)
4. [Design Patterns](#4-design-patterns)
5. [Project History](#5-project-history)

---

## 1. Architecture Overview

### 1.1 High-Level Module Hierarchy

```
src/
|
+-- trainers/                    # Training orchestration (entry point)
|   +-- trainer.py               # Base Trainer dataclass
|   +-- image_to_tag_trainers/   # Classification/tagging trainers
|   +-- image_to_image_trainers/ # AutoEncoder/U-Net trainers
|
+-- models/                      # Neural network wrappers
|   +-- model_wrapper.py         # Base ModelWrapper dataclass
|   +-- image_to_tag/            # Categorizer, Comparator wrappers
|   +-- image_to_image/          # AutoEncoder, Diffusion wrappers
|
+-- generators/                  # Data pipeline classes
|   +-- base_generators.py       # BatchGenerator, PostProcessGenerator
|   +-- image_to_tag/            # Classification, Comparator generators
|   +-- image_to_image/          # AutoEncoder, Foreground generators
|
+-- callbacks/                   # Keras callback implementations
|   +-- callback.py              # Base Callback dataclass
|   +-- plotters/                # Visualization callbacks
|       +-- image_to_tag/        # Confusion matrix, example plotters
|       +-- image_to_image/      # Reconstruction plotters
|
+-- losses/                      # Custom loss functions
|   +-- losses.py                # MAE, CrossEntropy, Dice, Edge loss
|   +-- from_model/              # Model-based losses (Blurriness, Encoding)
|
+-- modules/                     # Reusable neural network blocks
|   +-- module.py                # build_encoder(), build_decoder()
|   +-- blocks/                  # ConvolutionBlock, DeconvolutionBlock
|   +-- layers/                  # AtrousConv2D, PaddedConv2, ScaleLayer
|
+-- samples/                     # Data sample dataclasses
|   +-- sample.py                # Base Sample dataclass
|   +-- image_to_tag/            # ClassificationSample, ComparatorSample
|   +-- image_to_image/          # AutoEncodingSample, ForegroundSample
|
+-- output_spaces/               # Label/tag space management
|   +-- output_space.py          # Base OutputSpace dataclass
|   +-- tag.py                   # Tag dataclass
|   +-- custom/nested/           # Hierarchical categorization
```

### 1.2 Module Responsibilities

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| **trainers** | Orchestrates training: manages generators, model, callbacks, training loop | `Trainer`, `CategorizerTrainer`, `AutoEncoderTrainer` |
| **models** | Wraps Keras models with loss, metrics, and compilation logic | `ModelWrapper`, `CategorizerWrapper`, `AutoEncoderWrapper`, `Comparator` |
| **generators** | Produces input/output batches from filesystem data | `BatchGenerator`, `PostProcessGenerator`, `ClassificationGenerator` |
| **callbacks** | Provides training visualization and logging | `Callback`, `HistoryCallback`, `ExamplePlotter`, `ConfusionMatrixPlotter` |
| **losses** | Defines differentiable objective functions | `Loss`, `mae`, `cross_entropy`, `one_minus_dice`, `edge_loss` |
| **modules** | Provides reusable encoder-decoder architecture blocks | `build_encoder()`, `build_decoder()`, `ConvolutionBlock`, `DeconvolutionBlock` |
| **samples** | Encapsulates single data point with input/output processing | `Sample`, `ClassificationSample`, `AutoEncodingSample`, `ComparatorSample` |
| **output_spaces** | Manages label vocabularies and tag-to-index mappings | `OutputSpace`, `Tag`, `CategorizationSpace`, `ComparatorSpace` |

### 1.3 Module Dependency Graph

```
                         +---------------+
                         |    Trainer    |
                         +-------+-------+
                                 |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
  +---------------+      +---------------+      +---------------+
  |  ModelWrapper |      | BatchGenerator|      |   Callback    |
  +-------+-------+      +-------+-------+      +-------+-------+
          |                      |                      |
          |                      v                      |
          |              +---------------+              |
          |              |    Sample     |              |
          |              +-------+-------+              |
          |                      |                      |
          v                      v                      v
  +---------------+      +---------------+      +---------------+
  |    modules    |      | OutputSpace   |      | ModelWrapper  |
  | (encoder/dec) |      | (tag lookup)  |      | (for plotting)|
  +-------+-------+      +---------------+      +---------------+
          |
          v
  +---------------+
  |    losses     |
  +---------------+
```

### 1.4 Import Relationships (Detailed)

**Trainer imports:**
- `src.generators.base_generators.BatchGenerator`
- `src.generators.base_generators.PostProcessGenerator`
- `src.models.model_wrapper.ModelWrapper`
- `src.callbacks.history_callback.HistoryCallback`

**ModelWrapper imports:**
- `src.modules.module.build_encoder`
- `src.on_model_start.write_summary, backup`
- `src.config.LEARNING_RATE`

**BatchGenerator imports:**
- `src.output_spaces.output_space.OutputSpace`
- `src.samples.sample.Sample`

**Callback imports:**
- `src.models.model_wrapper.ModelWrapper`

**modules/module.py imports:**
- `src.modules.blocks.convolution_block.ConvolutionBlock`
- `src.modules.blocks.deconvolution_block.DeconvolutionBlock`

---

## 2. Data Flow Patterns

### 2.1 Image-to-Tag Pipeline

```
Filesystem (images + tags)
         |
         v
+------------------------+
| ClassificationGenerator| --> iterates over filenames
+------------------------+
         |
         | for each batch:
         v
+------------------------+
| ClassificationSample   | --> reads image, gets tags from OutputSpace
+------------------------+
         |
         | (inputs, outputs) = (image_array, one_hot_tags)
         v
+------------------------+
| PostProcessGenerator   | --> optional augmentation (e.g., VerticalSymmetry)
+------------------------+
         |
         v
+------------------------+
| CategorizerWrapper     | --> encoder -> dense layers -> sigmoid
+------------------------+
         |
         v
     Predictions (multi-label classification)
```

**Key Classes:**
- `ClassificationGenerator` (`src/generators/image_to_tag/classification_generator.py:10`)
- `CategorizerWrapper` (`src/models/image_to_tag/categorizer_wrapper.py:13`)
- `OutputSpace` (`src/output_spaces/output_space.py:13`)

### 2.2 Image-to-Image Pipeline

```
Filesystem (input images, optional target images)
         |
         v
+------------------------+
| AutoEncoderGenerator   | --> iterates over filenames
+------------------------+
         |
         v
+------------------------+
| AutoEncodingSample     | --> input = output (reconstruction task)
+------------------------+
         |
         v
+------------------------+
| PostProcessGenerator   | --> augmentation (symmetry, blur, etc.)
+------------------------+
         |
         v
+------------------------+
| AutoEncoderWrapper     | --> encoder -> decoder (U-Net style)
+------------------------+
         |
         v
     Reconstructed Image (same shape as input)
```

**Key Classes:**
- `AutoEncoderGenerator` (`src/generators/image_to_image/autoencoder_generator.py:5`)
- `AutoEncoderWrapper` (`src/models/image_to_image/auto_encoder_wrapper.py:14`)

### 2.3 Generator-Model-Loss Relationship

```
+----------------+     produces      +----------------+
|  BatchGenerator|  ------------->  | (inputs, outputs)|
+----------------+                   +----------------+
                                            |
                                            v
                                    +----------------+
                                    |  ModelWrapper  |
                                    |  .model.fit()  |
                                    +-------+--------+
                                            |
                  +-------------------------+-------------------------+
                  |                         |                         |
                  v                         v                         v
          +----------------+        +----------------+        +----------------+
          |  model(inputs) |        |  loss(y_true,  |        |  metrics       |
          |  -> y_pred     |        |       y_pred)  |        |  evaluation    |
          +----------------+        +----------------+        +----------------+
```

### 2.4 Sample Classes Data Handling

| Sample Class | Input Data | Output Data | Use Case |
|--------------|------------|-------------|----------|
| `Sample` (base) | Image from `input_filename` | Abstract | Base class |
| `ClassificationSample` | Image array | One-hot from OutputSpace | Multi-label tagging |
| `AutoEncodingSample` | Image array | Same as input | Reconstruction |
| `ComparatorSample` | (anchor, positive, negative) | Distance targets | Similarity learning |
| `ImageToImageSample` | Source image | Target image | Paired transformation |

---

## 3. Extension Points

### 3.1 Adding New Models

**Base class:** `ModelWrapper` (`src/models/model_wrapper.py:16`)

**Required overrides:**
```python
@dataclass
class CustomWrapper(ModelWrapper):
    @LazyProperty
    def loss(self) -> Callable:
        # Return loss function or Loss object
        return Loss([cross_entropy], class_weights=self.class_weights)

    @LazyProperty
    def metrics(self) -> Sequence[Callable]:
        # Return list of metric functions
        return [accuracy]

    @property
    def output_layer(self) -> tf.keras.layers.Layer:
        # Build and return model output layer(s)
        encoded = self.encoded_layer
        output = tf.keras.layers.Dense(self.output_shape[0], activation='sigmoid')(encoded)
        return output
```

**Examples:**
- `CategorizerWrapper` - Classification with BCE loss
- `AutoEncoderWrapper` - Reconstruction with MAE + edge loss
- `Comparator` - Siamese network with triplet architecture

### 3.2 Adding New Generators

**Base class:** `BatchGenerator` (`src/generators/base_generators.py:11`)

**Required overrides:**
```python
class CustomGenerator(BatchGenerator):
    def reader(self, input_filename: str) -> Sample:
        # Return a Sample subclass instance
        return CustomSample(input_filename, self.shape)
```

**For post-processing (augmentation):**
```python
@dataclass
class CustomPostProcessor(PostProcessGenerator):
    def __call__(self, inputs: np.ndarray, outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Transform batch
        return augmented_inputs, augmented_outputs
```

**Examples:**
- `ClassificationGenerator` - Tag-based classification
- `BlurryGenerator` - Adds blur augmentation
- `VerticalSymmetryGenerator` - Random horizontal flip

### 3.3 Adding New Loss Functions

**Pattern:** Function with signature `(y_true, y_pred, class_weights, epsilon) -> tf.Tensor`

```python
def custom_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    class_weights: Optional[tf.Tensor] = None,
    epsilon: float = 1e-7
) -> tf.Tensor:
    loss = ...  # Compute loss
    loss = Loss.apply_class_weights(loss, class_weights)
    return tf.reduce_mean(loss)
```

**Combining losses:**
```python
loss = Loss([mae, edge_loss, one_minus_dice], loss_weights=[1.0, 0.5, 0.3])
```

**Examples in `src/losses/losses.py`:**
- `mae` - Mean Absolute Error
- `cross_entropy` - Binary Cross Entropy
- `one_minus_dice` - Dice coefficient loss
- `edge_loss` - Sobel edge preservation

### 3.4 Adding New Callbacks

**Base class:** `Callback` (`src/callbacks/callback.py:11`)

```python
@dataclass
class CustomCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        # Access self.model (Keras model)
        # Access self.model_wrapper (ModelWrapper)
        # Access self.output_path for saving
        pass
```

**Examples:**
- `HistoryCallback` - Logs training history to CSV
- `ExamplePlotter` - Visualizes predictions periodically
- `ConfusionMatrixPlotter` - Plots classification confusion matrix

### 3.5 Adding New Custom Layers

**Base:** Extend `tf.keras.layers.Layer`

```python
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize weights
        self.kernel = self.add_weight(...)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Forward pass
        return tf.matmul(inputs, self.kernel)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(dict(units=self.units))
        return config
```

**Examples:**
- `AtrousConv2D` (`src/modules/layers/atrous_conv2d.py:9`) - Dilated convolution
- `PaddedConv2` - Convolution with explicit padding
- `ScaleLayer` (`src/modules/layers/scale_layer.py`) - Learnable scaling

---

## 4. Design Patterns

### 4.1 Factory Pattern - Model Wrapper Creation

**Location:** `Trainer.get_model_wrapper` property

The Trainer delegates model creation to a factory method that each subclass overrides:

```python
# In CategorizerTrainer
@property
def get_model_wrapper(self) -> Type:
    return CategorizerWrapper  # Factory returns class

# In Trainer.set_model_wrapper()
self._model_wrapper = self.get_model_wrapper(  # Instantiation
    self.name,
    self.input_shape,
    ...
)
```

### 4.2 Strategy Pattern - Loss Functions and Generators

**Losses as strategies:**
```python
# Different loss strategies can be composed
loss_strategy = Loss([mae, edge_loss], loss_weights=[1.0, 0.5])
model.compile(loss=loss_strategy)
```

**Generators as strategies:**
```python
# PostProcessGenerator chain applies different augmentation strategies
generator = compose_generators(base_generator, [
    VerticalSymmetryGenerator,
    BlurryGenerator,
])
```

### 4.3 Template Method Pattern - Trainer Classes

**Base:** `Trainer` defines the algorithm skeleton in `run()`:

```python
def run(self):
    self.model_wrapper.fit(
        self.model_wrapper.training_generator,
        batch_size=self.batch_size,
        validation_data=self.model_wrapper.test_generator,
        steps_per_epoch=self.training_steps,
        callbacks=self.callbacks,  # Hook
        ...
    )
```

**Hooks overridden by subclasses:**
- `get_model_wrapper` - Which model to use
- `callbacks` - Which callbacks to attach
- `post_process_generator_classes` - Which augmentations

### 4.4 Custom Patterns

**Nested Categorization:**
- `Category` class (`src/output_spaces/custom/nested/nested_tags.py:21`)
- Hierarchical tag structure with parent-child relationships
- Enables multi-level classification

**Comparator Architecture (Siamese Network):**
- `Comparator` class (`src/models/image_to_tag/comparator_wrapper.py:16`)
- Triplet input: (anchor, positive, negative)
- Shared encoder with distance-based loss

### 4.5 Coding Conventions

| Convention | Example | Location |
|------------|---------|----------|
| Dataclass for configuration | `@dataclass class Trainer:` | All base classes |
| LazyProperty for expensive init | `@LazyProperty def model:` | ModelWrapper, Trainer |
| Type hints on all signatures | `def loss(self) -> Callable:` | Throughout |
| Private underscore prefix | `_model`, `_filenames` | Backing fields |
| Wrapper suffix for model classes | `CategorizerWrapper` | `src/models/` |
| Generator suffix for data classes | `ClassificationGenerator` | `src/generators/` |

---

## 5. Project History

### 5.1 Initial Purpose and Goals

The ML Framework was designed to provide:
1. **Modular architecture** for image-based deep learning tasks
2. **Separation of concerns** between data loading, model definition, and training
3. **Extensibility** through base classes and composition patterns
4. **Visualization support** via callback system

### 5.2 Key Model Architectures Implemented

| Architecture | Wrapper Class | Description |
|--------------|---------------|-------------|
| **U-Net** | `AutoEncoderWrapper` | Encoder-decoder with skip connections |
| **Classifier** | `CategorizerWrapper` | Encoder + dense layers for multi-label |
| **Nested Classifier** | `CategorizerWrapper` (nested) | Hierarchical categorization |
| **Comparator (Siamese)** | `Comparator` | Triplet-based similarity learning |
| **Diffusion** | `DiffusionModelWrapper` | Noise prediction for generation |

### 5.3 Current Capabilities

- **Image-to-Tag**: Multi-label classification, hierarchical tagging, rating prediction
- **Image-to-Image**: Reconstruction, colorization, segmentation, saliency detection
- **Training Features**: Data augmentation, class weighting, callback visualization
- **Custom Layers**: Dilated convolutions, residual blocks, scale layers

---

## References

- [README.md](../../README.md) - Installation and basic usage
- [index-codebase.md](../indices/index-codebase.md) - 411-entry keyword-to-location lookup
- [index-lessons-learned.md](../indices/index-lessons-learned.md) - Common pitfalls and solutions

---

*Document created as part of Task X.4 - Project Overview*
