# 005: The clean-clone test earns its keep

Date: 2026-06-12

## What happened

After everything was green locally (all gates, all results), an actual
clean-clone test (git clone to a temp dir, uv sync, run the gates) failed
9 of 18 tests with "pygame.error: File is not a Windows BMP file".

The dev venv runs Python 3.13. The project never pinned an interpreter,
so the fresh clone resolved to Python 3.14, where the pygame 2.6.1 build
ships without extended image support (pygame.image.get_extended() is
False) and silently cannot load PNGs. Locked package versions do not pin
the interpreter.

## Fix

- requires-python capped to >=3.12,<3.14 (pygame 2.6.1 does not properly
  support 3.14).
- .python-version (3.13) committed; it was previously gitignored under a
  pyenv-era rule, which is exactly how an interpreter pin goes missing.
- A fail-fast ImportError with a clear message if a pygame build without
  extended image support ever loads the assets module.

## Lesson

"Reproducible from a clean clone" is a claim about an experiment you have
to actually run, like any other. The lock file pinned every package and
still let the environment drift on the one dependency outside the lock:
the interpreter. Run the clean-clone test on the real artifact, from a
different directory, before calling reproducibility done.
