# Enhancement Proposal Plan (EPP)

This directory manages the enhancement workflow for the **ML Framework** project - an Image-to-Image and Image-to-Tag deep learning framework built on TensorFlow.

## Project Overview

- **Repository**: ML Framework
- **Domain**: Deep Learning, Computer Vision
- **Core Technologies**: TensorFlow/Keras, Python 3.12+
- **EPP Initialized**: 2025-12-25

## Workflow Hierarchy

```
Phase > Task > Step > Atomic Action
```

- **Phase**: Major development milestone (directory in `phases/`)
- **Task**: Discrete work unit within a Phase (markdown file)
- **Step**: Sequence of atomic actions within a Task
- **Atomic Action**: Smallest unit of development

## Directory Structure

```
.ENHANCEMENT_PROPOSAL_PLAN/
├── README.md                    # This file
├── phases/                      # Phase directories
│   └── phase-X-off-chronology/  # Ongoing/maintenance tasks
├── indices/                     # Lookup tables (keyword -> reference)
│   ├── index-codebase.md        # keyword -> file:line
│   ├── index-documentation.md   # keyword -> doc location
│   ├── index-lessons-learned.md # keywords -> lesson file
│   ├── index-phases.md          # phase/task -> status
│   ├── index-repositories.md    # repo -> path
│   └── index-agents.md          # agent -> responsibility
├── lessons-learned/             # Individual lesson files
└── questions/                   # Pending user clarifications
```

## Governance

- **Archivist Agent**: Sole editor of EPP directory
- **Round Table Discussions (RTD)**: Convened at Phase/Task start for planning
- **Index Maintenance**: All agents must consult indices before implementation

## Status Legend

| Status | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked / Needs attention |
