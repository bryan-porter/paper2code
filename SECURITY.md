# Security Policy

## Supported code

Security fixes are made on the `main` branch. Use the latest revision of `main`
when evaluating or reporting an issue.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or exposed
credential. Use GitHub's private vulnerability reporting for this repository. If
that feature is unavailable, contact the maintainer through an existing private
channel and include:

- the affected revision and file;
- the security impact and conditions required to reproduce it;
- a minimal reproduction that contains no real credentials or private data; and
- any suggested mitigation.

Do not include live secrets, private papers, datasets, or production output in a
report. If a real credential may have been exposed, revoke or rotate it before
sharing diagnostic details.

## Security scope

Papers, extracted text, linked repositories, generated code, model checkpoints,
notebooks, and package instructions are untrusted external content. The project
does not automatically install, clone, or execute them. Downloaded PDFs remain
inert and are never automatically opened or parsed. Run generated artifacts only
in a disposable environment without production credentials or access to
sensitive data.

The security-supported runtime is 64-bit Windows with CPython 3.13 and packages
from the canonical PyPI index. Install complete `requirements-*-win-py313.lock`
closures with `--require-hashes`; `.in` files are review inputs, not locks. Other
platforms are not covered by a reviewed dependency closure.

Output paths are created privately. POSIX systems additionally use
handle-relative, no-follow filesystem operations where Python exposes them. On
Windows, reparse/regular-file and identity checks cannot fully exclude a path-swap
race by another process running as the same principal. Use an ACL-isolated
directory and do not run untrusted same-account processes concurrently.
