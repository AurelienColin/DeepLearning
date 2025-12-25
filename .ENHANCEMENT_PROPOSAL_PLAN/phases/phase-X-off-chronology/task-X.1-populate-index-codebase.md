# Task X.1: Populate Index-Codebase

**Phase**: X - Off-Chronology
**Status**: `[ ]` Not Started
**Priority**: High
**Assigned Agent**: `python-pro` (with `archivist` for index updates)

---

## Objective

Populate `indices/index-codebase.md` with a searchable lookup table of all classes, functions, and methods in the codebase.

## Output Format

The index is a flat list of entries:

```markdown
| Keyword | Location | Type |
|---------|----------|------|
| UNetTrainer | src/trainers/image_to_image_trainers/unet_trainer.py:15 | class |
| kid_loss | src/losses/kid.py:42 | function |
```

## Steps

### Step 1: Extract Class Definitions
**Agent**: `python-pro`
**Status**: `[ ]`
- [ ] Run `grep -rnH "^class " src/ --include="*.py"`
- [ ] Format output as `ClassName | file:line | class`

### Step 2: Extract Function Definitions
**Agent**: `python-pro`
**Status**: `[ ]`
- [ ] Run `grep -rnH "^def " src/ --include="*.py"`
- [ ] Format output as `function_name | file:line | function`

### Step 3: Extract Method Definitions (public only)
**Agent**: `python-pro`
**Status**: `[ ]`
- [ ] Run `grep -rnH "    def [^_]" src/ --include="*.py"`
- [ ] Format output as `ClassName.method_name | file:line | method`

### Step 4: Populate Index
**Agent**: `archivist`
**Status**: `[ ]`
- [ ] Merge all entries into `indices/index-codebase.md`
- [ ] Sort alphabetically by keyword
- [ ] Remove duplicates

## Acceptance Criteria

- Every public class/function/method has an entry
- Line numbers are accurate
- Index is alphabetically sorted
- No prose, only table entries
