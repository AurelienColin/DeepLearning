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

*Last Updated: 2025-12-25*
