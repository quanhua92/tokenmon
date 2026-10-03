# Maintainer guide

The `CI` workflow runs on pull requests and pushes to `main`. It tests Python
3.10–3.14 on Linux and 3.13 on macOS, builds a wheel and source distribution, checks
package metadata and README rendering, and checks the installed CLI. Jobs use hosted
runners, read-only permissions, and actions pinned to full commit SHAs.

The `Publish to PyPI` workflow is manual and only runs from `main`. It validates the
requested version, runs tests, builds and checks distributions, then passes them to
a separate publishing job. Only the publishing job has OIDC permissions. It uses
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
without a stored API token.

One-time publisher setup:

1. In GitHub repository **Settings → Environments**, create `pypi`. Allow deployments
   from `main` only and configure a required reviewer for release approval.
2. On PyPI, add a GitHub trusted publisher with owner `quanhua92`, repository
   `tokenmon`, workflow filename `release.yml`, and environment `pypi`. For a new
   project, register a [pending publisher](https://pypi.org/manage/account/publishing/)
   with project name `tokenmon`; for an existing project, use its Publishing settings.

To release:

1. Commit the desired version in both `pyproject.toml` and
   `src/tokenmon/__init__.py`, refresh `uv.lock` with `uv lock`, and merge to `main`
   after CI passes. The workflow checks versions; it does not change them.
2. Open **Actions → Publish to PyPI → Run workflow**, select `main`, and enter the
   exact version, such as `0.1.1`.
3. Review the built distributions in the workflow artifact, then approve the `pypi`
   deployment. PyPI rejects uploading an already published distribution again.

Validate workflow edits locally with `actionlint`. Packaging tools are pinned in
`.github/requirements-build.txt`; they are separate from application dependencies.

## Updating the PyPI description

Edit `description` in `pyproject.toml` for the short summary, or `README.md` for the
full project description (`readme = "README.md"`). The build embeds these in the
distribution metadata. To update PyPI, publish a new version through the manual
release workflow, following the steps above. Pushing README changes or creating a
GitHub release alone does not update PyPI; published release metadata is immutable.
