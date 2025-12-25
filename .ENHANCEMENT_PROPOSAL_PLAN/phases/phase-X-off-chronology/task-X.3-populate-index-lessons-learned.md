# Task X.3: Populate Index-Lessons-Learned

**Phase**: X - Off-Chronology
**Status**: `[x]` Complete
**Priority**: Medium
**Assigned Agent**: `archivist` (with input from all agents)

---

## Objective

1. Create a `lessons-learned/` folder to store individual lesson files
2. Populate `indices/index-lessons-learned.md` with a lookup table referencing those files

## Directory Structure

```
.ENHANCEMENT_PROPOSAL_PLAN/
├── lessons-learned/
│   ├── import-circular-dependency.md
│   ├── test-gpu-memory.md
│   └── ci-tensorflow-version.md
└── indices/
    └── index-lessons-learned.md   # References above files
```

## Lesson File Format

Each lesson is a standalone markdown file:

```markdown
# [Short Title]

**Keywords**: keyword1, keyword2
**Related Commits**: abc1234, def5678

## Problem
[Brief description of the issue]

## Resolution
[How it was fixed]

## Prevention
[How to avoid in future]
```

## Index Format

```markdown
| Keywords | Lesson | Summary |
|----------|--------|---------|
| circular, import | lessons-learned/import-circular-dependency.md | Resolve via lazy import |
| GPU, memory, OOM | lessons-learned/test-gpu-memory.md | Reduce batch size in tests |
```

## Steps

### Step 1: Create Lessons Folder
**Agent**: `archivist`
**Status**: `[x]` (Folder exists)
- [x] Create `lessons-learned/` directory in EPP

### Step 2: Extract Lessons from Git History
**Agent**: `devops-engineer`
**Status**: `[x]`
- [x] Filter commits with "fix", "bug", "issue" keywords
- [x] For each significant fix, create a lesson file
- [x] Use commit message + diff to populate Problem/Resolution

**Created 5 lesson files:**
- `wbce-log-negative-values.md` (commit 97b4210)
- `division-by-zero-normalization.md` (commit c9f1b38)
- `ci-branch-naming.md` (commit 9ff2a9a)
- `hardcoded-file-extension.md` (commit 8bb9a64)
- `copy-paste-variable-error.md` (commit 2e34b02)

### Step 3: Populate Index
**Agent**: `archivist`
**Status**: `[x]`
- [x] Add entry for each lesson file
- [x] Keywords should be searchable terms
- [x] Summary is one-line

## Acceptance Criteria

- Each lesson is a separate file in `lessons-learned/`
- Index references all lesson files
- Keywords enable quick search
- No prose in index, only table entries
