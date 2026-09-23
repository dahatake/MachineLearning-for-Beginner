# Setup test maintenance

Run the static checks from the repository root:

```powershell
pwsh -NoProfile -File .\setup\tests\run-static-checks.ps1
```

`MLFB_TEST_ISOLATE=1` is reserved for CI and makes setup tests use fresh,
temporary install locations. Do not use it for a normal learner setup.

The full operating-system and installer matrix runs in GitHub Actions because
this repository's Windows development environment cannot execute macOS jobs.
