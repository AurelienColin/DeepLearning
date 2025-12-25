# Task X.3: Populate Index-Lessons-Learned

**Phase**: X - Off-Chronology
**Status**: `[ ]` Not Started
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
- [ ] Create `lessons-learned/` directory in EPP

### Step 2: Extract Lessons from Git History
- [ ] Filter commits with "fix", "bug", "issue" keywords
- [ ] For each significant fix, create a lesson file
- [ ] Use commit message + diff to populate Problem/Resolution

### Step 3: Populate Index
- [ ] Add entry for each lesson file
- [ ] Keywords should be searchable terms
- [ ] Summary is one-line

## Acceptance Criteria

- Each lesson is a separate file in `lessons-learned/`
- Index references all lesson files
- Keywords enable quick search
- No prose in index, only table entries
