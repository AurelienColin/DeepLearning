# Agents Index

> Agent responsibilities and when to invoke them

**Last Updated**: 2025-12-25

---

## Available Agents

| Agent | Responsibility | Typical Tasks |
|-------|---------------|---------------|
| `machine-learning-researcher` | ML architecture, loss functions, SOTA | Model design, training strategy |
| `python-pro` | Type safety, async, performance | Code implementation, optimization |
| `refactoring-specialist` | Clean code, complexity reduction | Code improvement, debt reduction |
| `technical-writer` | API docs, user guides | Documentation creation |
| `test-automator` | Pytest suites, mocks | Test coverage, CI integration |
| `agent-organizer` | Task decomposition | Planning complex tasks |
| `devops-engineer` | GitLab CI, Docker | CI/CD, containerization |
| `archivist` | EPP management | Index maintenance, RTD logging |
| `code-reviewer` | Quality, security | Code reviews, best practices |

## Agent Selection Rules

1. Match task to most specific agent
2. Explicitly announce context switch: *"Switching context to [Agent Name]..."*
3. `archivist` is sole editor of `.ENHANCEMENT_PROPOSAL_PLAN/`
4. Consult indices before implementation

## Round Table Discussions (RTD)

- Triggered at Phase/Task start
- Any agent can request via `archivist`
- Purpose: Review plan, raise concerns, identify dependencies

---

*Maintained by: archivist agent*
