# Task X.1: Populate Index-Codebase

**Phase**: X - Off-Chronology
**Status**: `[ ]` Not Started
**Priority**: High
**Assigned Agent**: `python-pro` (with `archivist` for index updates)

---

## Objective

Create a comprehensive index mapping keywords to code locations (class/function -> file:line) to facilitate future navigation and development.

## Scope

Map the following modules:
- Models (`src/models/`)
- Generators (`src/generators/`)
- Losses (`src/losses/`)
- Trainers (`src/trainers/`)
- Modules/Layers (`src/modules/`)
- Output Spaces (`src/output_spaces/`)
- Callbacks (`src/callbacks/`)
- Samples (`src/samples/`)

## Steps

### Step 1: Inventory Models
- [ ] List all model classes in `src/models/`
- [ ] Document class name, file path, line number, and purpose
- [ ] Identify inheritance hierarchy

### Step 2: Inventory Generators
- [ ] List all generator classes in `src/generators/`
- [ ] Document data flow (input -> output)
- [ ] Note image_to_tag vs image_to_image variants

### Step 3: Inventory Losses
- [ ] List all loss functions in `src/losses/`
- [ ] Document mathematical purpose
- [ ] Note model-dependent losses (`from_model/`)

### Step 4: Inventory Trainers
- [ ] List all trainer classes in `src/trainers/`
- [ ] Document training workflows
- [ ] Note benchmark scripts

### Step 5: Inventory Modules
- [ ] List all layers in `src/modules/layers/`
- [ ] List all blocks in `src/modules/blocks/`
- [ ] Document custom objects

### Step 6: Inventory Output Spaces
- [ ] List output space definitions
- [ ] Document tag handling
- [ ] Note custom output spaces

### Step 7: Inventory Callbacks
- [ ] List all callback classes
- [ ] Document plotter callbacks
- [ ] Note image_to_tag vs image_to_image variants

### Step 8: Update Index
- [ ] Populate `indices/index-codebase.md` with findings
- [ ] Verify all entries have correct line numbers
- [ ] Cross-reference with related indices

## Acceptance Criteria

- All public classes/functions are indexed
- Line numbers are accurate
- Descriptions are concise and informative
- Index is formatted consistently

## Notes

- Use `grep -rn "class " src/` to find class definitions
- Use `grep -rn "def " src/` to find function definitions
- Verify line numbers after any refactoring
