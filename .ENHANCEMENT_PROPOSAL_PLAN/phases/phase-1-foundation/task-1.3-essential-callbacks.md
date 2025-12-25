# Task 1.3: Essential Callbacks

**Phase:** 1 - Foundation & Infrastructure
**Status:** `[x]` Complete
**Priority:** High
**Assigned Agent:** `python-pro`

---

## Objective

Add essential training callbacks that prevent training loss and enable early stopping on performance plateau. These are standard ML best practices currently missing from the framework.

## Scope

Address items from Task 1.0 (Assessment):
- **FG-035:** Add ModelCheckpoint callback
- **FG-036:** Add EarlyStopping callback

---

## Implementation Steps

### Step 1.3.1: Integrate ModelCheckpoint Callback

**Status:** `[x]` Complete
**Agent:** `python-pro`
**Effort:** S (< 30 min)

**Purpose:** Save model weights periodically and/or when validation loss improves

**Implementation Options:**

**Option A: Use Keras Built-in (Recommended)**
```python
# In trainer setup
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(output_path, 'model_{epoch:02d}_{val_loss:.4f}.keras'),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)
```

**Option B: Wrap in Custom Callback**
```python
# src/callbacks/checkpoint_callback.py
from dataclasses import dataclass
from tensorflow.keras.callbacks import ModelCheckpoint
from src.callbacks.callback import Callback

@dataclass
class CheckpointCallback(Callback):
    """Wrapper around Keras ModelCheckpoint for framework integration."""
    monitor: str = 'val_loss'
    save_best_only: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._keras_callback = ModelCheckpoint(
            filepath=os.path.join(self.output_path, 'model_best.keras'),
            monitor=self.monitor,
            save_best_only=self.save_best_only,
            mode='min',
            verbose=1
        )

    def set_model(self, model):
        super().set_model(model)
        self._keras_callback.set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        self._keras_callback.on_epoch_end(epoch, logs)
```

**File Format Note:**
- Use `.keras` format (not `.h5`) per TD-001 through TD-008 recommendations
- Aligns with TensorFlow 2.13+ best practices

**Integration Point:**
- Add to `Trainer.callbacks` property
- Make configurable via `Trainer` dataclass field

**User answer:** I accept Option A.


---

### Step 1.3.2: Integrate EarlyStopping Callback

**Status:** `[x]` Complete
**Agent:** `python-pro`
**Effort:** S (< 30 min)

**Purpose:** Stop training when validation loss stops improving

**Implementation:**
```python
# In trainer setup
from tensorflow.keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,  # epochs to wait
    min_delta=1e-4,  # minimum improvement threshold
    restore_best_weights=True,
    verbose=1
)
```

**Configuration Options:**
```python
@dataclass
class Trainer:
    # Existing fields...
    early_stopping_patience: int = 10  # 0 to disable
    checkpoint_enabled: bool = True
```

---

### Step 1.3.3: Update Trainer Base Class

**Status:** `[x]` Complete
**Agent:** `python-pro`
**Effort:** S (< 30 min)

**Location:** `src/trainers/trainer.py`

**Changes Required:**

1. Add configuration fields:
```python
@dataclass
class Trainer:
    # ... existing fields ...
    checkpoint_enabled: bool = True
    early_stopping_patience: int = 10  # 0 to disable
```

2. Update callbacks property:
```python
@property
def callbacks(self) -> list:
    cbs = [self.history_callback]

    if self.checkpoint_enabled:
        cbs.append(ModelCheckpoint(
            filepath=os.path.join(self.output_path, 'model_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ))

    if self.early_stopping_patience > 0:
        cbs.append(EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True
        ))

    return cbs + self.custom_callbacks
```

---

## Acceptance Criteria

- [x] ModelCheckpoint saves `.keras` files (not `.h5`)
- [x] EarlyStopping triggers correctly when validation loss plateaus
- [x] Both callbacks are configurable (enable/disable, parameters)
- [x] `pytest tests/` passes (pre-existing failures unrelated to this task)
- [x] Training saves checkpoint when `checkpoint_enabled=True`
- [x] Changes committed with message: `[Feat] Add ModelCheckpoint and EarlyStopping callbacks`

---

## Notes

- Prefer Keras built-ins over custom implementations for maintainability
- Ensure checkpoint path uses `output_path` from Trainer
- Test with short training run to verify behavior

---

## Cross-References

- **Source:** [Task 1.0 - Assessment](task-1.0-assessment.md)
- **Backlog:** [improvement-backlog.md §8.5](../../specs/improvement-backlog.md#85-training-infrastructure-gaps)
- **Extension Points:** [project-overview.md §3.4](../../specs/project-overview.md#34-adding-new-callbacks)

---

*Created: 2025-12-25 via Phase 1 RTD*
