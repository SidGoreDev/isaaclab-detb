# Contributing

## Workflow

- work from small, reviewable changes
- keep the `mock` path passing while developing new simulator-backed features
- keep simulator-specific logic behind backend seams
- update docs when behavior, artifacts, or config semantics change
- add tests for every new command or persisted contract

## Pull Request Expectations

- explain the user-facing or artifact-facing change
- note any config contract changes explicitly
- include or update tests
- mention whether the change affects `mock`, `isaaclab`, or both

## Commit Guidance

- prefer direct commit messages
- keep commits scoped to one coherent change
- do not include transient outputs or secrets
