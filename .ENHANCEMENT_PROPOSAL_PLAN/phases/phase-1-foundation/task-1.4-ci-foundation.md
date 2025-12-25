# Task 1.4: CI/CD Foundation

**Phase:** 1 - Foundation & Infrastructure
**Status:** `[ ]` Not Started
**Priority:** Medium
**Assigned Agent:** `devops-engineer`
**Depends On:** Task 1.1 (Critical Fixes)

---

## Objective

Establish automated code quality gates using pre-commit hooks to prevent regressions and enforce coding standards.

## Scope

Address items from Task 1.0 (Assessment):
- **TD-029:** Add pre-commit hooks configuration

---

## Implementation Steps

### Step 1.4.1: Create Pre-commit Configuration

**Status:** `[ ]`
**Agent:** `devops-engineer`
**Effort:** S (< 30 min)

**File:** `.pre-commit-config.yaml` (project root)

**Content:**
```yaml
# .pre-commit-config.yaml
# See https://pre-commit.com for more information

repos:
  # Standard pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: debug-statements

  # Python code formatting check (not auto-format)
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args:
          - '--max-line-length=100'
          - '--extend-ignore=E203,E501,W503'
          - '--select=F,E9,W'  # Errors only, not style
        exclude: ^(\.tmp/|data/)

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--check-only', '--diff']

# Optional: Enable for stricter checks
#  - repo: https://github.com/pre-commit/mirrors-mypy
#    rev: v1.8.0
#    hooks:
#      - id: mypy
#        args: ['--ignore-missing-imports']
#        additional_dependencies: [tensorflow-stubs]
```

**Rationale:**
- `check-added-large-files`: Prevents accidental commit of data files
- `flake8` with `--select=F,E9,W`: Catches errors, not style (style can be fixed incrementally)
- `isort --check-only`: Reports issues without auto-fixing (to preserve intentional ordering)
- `mypy` commented out: Enable after type hint cleanup in Phase 2

---

### Step 1.4.2: Update .gitignore

**Status:** `[ ]`
**Agent:** `devops-engineer`
**Effort:** S (< 15 min)

**Verify/Add patterns:**
```gitignore
# Data files (NEVER commit)
*.nc
*.h5
*.hdf*
*.grib*
*.SAFE/
*.tar.gz
*.zip

# Model weights
*.keras
*.ckpt*
*.pb

# Temporary
.tmp/
__pycache__/
*.pyc
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
```

---

### Step 1.4.3: Install and Test Pre-commit

**Status:** `[ ]`
**Agent:** `devops-engineer`
**Effort:** S (< 15 min)

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

**Verification:**
```bash
# Run on all files to check current state
pre-commit run --all-files

# Expected: Some failures on existing code (known from Task 1.0)
# After Task 1.1: Should pass
```

**Add to setup.py/pyproject.toml (dev dependencies):**
```toml
[project.optional-dependencies]
dev = [
    "pre-commit>=3.0",
    "flake8>=7.0",
    "isort>=5.0",
]
```

---

### Step 1.4.4: Document Pre-commit Usage

**Status:** `[ ]`
**Agent:** `devops-engineer`
**Effort:** S (< 15 min)

**Add to README.md:**
```markdown
## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality. To set up:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

To skip hooks temporarily (not recommended):

```bash
git commit --no-verify
```
```

---

## Acceptance Criteria

- [ ] `.pre-commit-config.yaml` exists and is valid
- [ ] `.gitignore` includes data file patterns
- [ ] `pre-commit run --all-files` passes after Task 1.1 completion
- [ ] `pre-commit install` works without errors
- [ ] README documents pre-commit usage
- [ ] Changes committed with message: `[DevOps] Add pre-commit hooks for code quality`

---

## Notes

- **Dependency:** Wait for Task 1.1 to complete before running `pre-commit run --all-files`
- Start with lenient rules; tighten incrementally in Phase 2
- Consider GitLab CI integration in future phase

---

## Cross-References

- **Source:** [Task 1.0 - Assessment](task-1.0-assessment.md)
- **Backlog:** [improvement-backlog.md §9.5](../../specs/improvement-backlog.md#95-cicd-debt)
- **Lessons:** [ci-branch-naming.md](../../lessons-learned/ci-branch-naming.md)

---

*Created: 2025-12-25 via Phase 1 RTD*
