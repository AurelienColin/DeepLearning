# Task X.2: Populate Index-Documentation

**Phase**: X - Off-Chronology
**Status**: `[ ]` Not Started
**Priority**: High
**Assigned Agent**: `technical-writer` (with `archivist` for index updates)

---

## Objective

Populate `indices/index-documentation.md` with a searchable lookup table referencing EPP documents and other documentation files.

## Output Format

The index is a flat list of entries:

```markdown
| Keyword | Location | Description |
|---------|----------|-------------|
| EPP workflow | .ENHANCEMENT_PROPOSAL_PLAN/README.md | Hierarchy and governance |
| Phase X | phases/phase-X-off-chronology/README.md | Off-chronology tasks |
| UNet training | src/trainers/.../run/README.md | Usage example |
```

## Steps

### Step 1: Index EPP Structure
- [ ] List all markdown files in `.ENHANCEMENT_PROPOSAL_PLAN/`
- [ ] Add entry for each with keyword and brief description

### Step 2: Index Project Documentation
- [ ] List README files in `src/` directories
- [ ] Add entry for each with keyword and brief description

### Step 3: Index Configuration/Examples
- [ ] Identify any config files or example scripts
- [ ] Add entry for each with keyword and brief description

### Step 4: Populate Index
- [ ] Merge all entries into `indices/index-documentation.md`
- [ ] Sort alphabetically by keyword

## Acceptance Criteria

- Every documentation file has an entry
- Keywords are searchable terms
- Descriptions are one-line summaries
- No prose, only table entries
