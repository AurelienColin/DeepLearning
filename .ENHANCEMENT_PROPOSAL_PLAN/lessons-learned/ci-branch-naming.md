# CI Branch Naming Mismatch

**Keywords**: CI, GitHub Actions, branch, main, master, workflow
**Related Commits**: 9ff2a9a

## Problem

GitHub Actions workflow was configured to trigger on `main` branch, but the repository's default branch was named `master`. This caused CI to not run on pushes to the actual default branch.

```yaml
# Problematic configuration
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

## Resolution

Update the workflow file to use the correct branch name:

```yaml
# Fixed configuration
on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]
```

## Prevention

- Verify branch naming conventions when setting up CI/CD
- When copying workflow templates, always check branch names match repository settings
- Consider using `branches: [ $default-branch ]` or explicit configuration that references actual branch names
- Document default branch naming convention in repository README or CONTRIBUTING.md
