# Task 1.1: Critical Code Fixes

**Phase:** 1 - Foundation & Infrastructure
**Status:** `[ ]` Not Started
**Priority:** Critical
**Assigned Agent:** `python-pro`

---

## Objective

Fix undefined name errors and critical type issues that could cause runtime failures or prevent code from executing correctly.

## Scope

Address items identified in Task 1.0 (Assessment):
- **IMP-001:** Fix undefined names (`OutputSpace`, `typing`)
- **IMP-002:** Add None guards to generator/iterator access

---

## Implementation Steps

### Step 1.1.1: Fix Undefined Name - OutputSpace

**Status:** `[ ]`
**Agent:** `python-pro`
**Effort:** S (< 15 min)

**Location:** `src/generators/image_to_image/overlay_generator.py:30`
**Issue:** Flake8 F821 - Undefined name 'OutputSpace'

**Action:**
```python
# Add import at top of file
from src.output_spaces.output_space import OutputSpace
```

**Verification:**
```bash
flake8 src/generators/image_to_image/overlay_generator.py --select=F821
```

**Codebase Reference:**
- [index-codebase.md](../../indices/index-codebase.md) - OutputSpace -> `src/output_spaces/output_space.py:13`

---

### Step 1.1.2: Fix Undefined Name - typing

**Status:** `[ ]`
**Agent:** `python-pro`
**Effort:** S (< 15 min)

**Location:** `src/generators/normalizer.py:25`
**Issue:** Flake8 F821 - Undefined name 'typing'

**Action:**
```python
# Add import at top of file
import typing
# OR if specific type needed:
from typing import <SpecificType>
```

**Verification:**
```bash
flake8 src/generators/normalizer.py --select=F821
```

---

### Step 1.1.3: Add None Guards to Generators

**Status:** `[ ]`
**Agent:** `python-pro`
**Effort:** S (< 30 min)

**Locations:** (from mypy analysis in Task 1.0)
- `src/generators/base_generators.py:58-72`
- `src/output_spaces/output_space.py:44-59`

**Pattern:**
```python
# Before
value = self.iterator.next()  # Could be None

# After
value = self.iterator.next()
if value is None:
    raise StopIteration("Generator exhausted")
```

**Verification:**
```bash
mypy --ignore-missing-imports src/generators/base_generators.py
```

---

## Acceptance Criteria

- [ ] `flake8 src/ --select=F821` returns no errors
- [ ] `mypy --ignore-missing-imports src/generators/` shows reduced error count
- [ ] `pytest tests/` passes (no regressions)
- [ ] Changes committed with message: `[Fix] Critical: Resolve undefined names and None guards`

---

## Notes

- These fixes are prerequisites for Task 1.4 (CI/CD Foundation)
- Pre-commit hooks cannot be enabled until these issues are resolved
- Keep changes minimal; do not refactor surrounding code

---

## Cross-References

- **Source:** [Task 1.0 - Assessment](task-1.0-assessment.md)
- **Backlog:** [improvement-backlog.md §5](../../specs/improvement-backlog.md#5-prioritized-improvement-queue)
- **Lessons:** [copy-paste-variable-error.md](../../lessons-learned/copy-paste-variable-error.md)

---

*Created: 2025-12-25 via Phase 1 RTD*
