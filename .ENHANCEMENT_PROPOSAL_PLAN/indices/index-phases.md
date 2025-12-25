# Phases Index

> Lookup table: Phase/Task -> status

---

## Active Phases

### Phase 1: Foundation & Infrastructure

| Task | Title | Status | Agent |
|------|-------|--------|-------|
| 1 | Foundation & Infrastructure | `[~]` In Progress | `agent-organizer` |
| 1.0 | Project Assessment | `[x]` | `agent-organizer` |
| 1.1 | Critical Code Fixes | `[ ]` | `python-pro` |
| 1.2 | Performance Quick Wins | `[ ]` | `machine-learning-researcher` |
| 1.3 | Essential Callbacks | `[ ]` | `python-pro` |
| 1.4 | CI/CD Foundation | `[ ]` | `devops-engineer` |

**Location:** [phases/phase-1-foundation/](../phases/phase-1-foundation/)

---

### Phase X: Off-Chronology

| Task | Title | Status | Notes |
|------|-------|--------|-------|
| X | Off-Chronology | Active | Maintenance & ad-hoc tasks |
| X.1 | Populate index-codebase | `[x]` | |
| X.2 | Populate index-documentation | `[x]` | |
| X.3 | Populate index-lessons-learned | `[x]` | |
| X.4 | Create project overview | `[x]` | |
| X.5 | Document potential improvements | `[x]` | *Retroactively moved to Task 1.0* |

**Location:** [phases/phase-X-off-chronology/](../phases/phase-X-off-chronology/)

---

## Future Phases (Proposed Roadmap)

Based on the improvement backlog analysis (Task 1.0 / X.5.6):

| Phase ID | Name | Effort | Priority | Depends On |
|----------|------|--------|----------|------------|
| **1** | Foundation & Infrastructure | S | Critical | - |
| 2 | Type Safety & Modernization | M | High | Phase 1 |
| 3 | Performance Optimization | M | High | Phase 1 |
| 4 | Core Features - Losses & Metrics | M | High | Phase 1 |
| 5 | Modern Backbones | M-L | Medium | Phase 4 |
| 6 | Documentation | M | Medium | Phase 2 |
| 7 | Advanced Architectures | L | Low | Phase 5 |
| 8 | Training Ecosystem | M | Low | Phase 3 |

**Dependency Graph:**
```
Phase 1 (Foundation) ─┬─> Phase 2 (Type Safety) ──> Phase 6 (Docs)
                      │
                      ├─> Phase 3 (Performance) ──> Phase 8 (Ecosystem)
                      │
                      └─> Phase 4 (Losses/Metrics) ──> Phase 5 (Backbones) ──> Phase 7 (Architectures)
```

**Reference:** [improvement-backlog.md §11.2](../specs/improvement-backlog.md#112-unified-phase-roadmap)

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked / Needs attention |

---

*Last Updated: 2025-12-25 (Phase 1 created via RTD)*
