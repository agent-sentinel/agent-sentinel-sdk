# AgentSentinel SDK - Release Guide

Releases are tag-driven. Pushing a `sdk-vX.Y.Z` tag runs
`.github/workflows/release-sdk.yml`, which builds the package, checks it,
publishes to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(no API token stored in the repo), and mirrors the source to the public
`agent-sentinel/agent-sentinel-sdk` repo.

---

## One-time setup: PyPI trusted publisher

Before the first tag-driven release, a PyPI project owner configures a
trusted publisher at https://pypi.org/manage/project/agentsentinel-sdk/settings/publishing/
(or, for a brand-new project, at https://pypi.org/manage/account/publishing/)
with:

| Field | Value |
| --- | --- |
| PyPI Project Name | `agentsentinel-sdk` |
| Owner | `agent-sentinel` |
| Repository name | `agent-sentinel` |
| Workflow name | `release-sdk.yml` |
| Environment name | `pypi` |

This lets the `publish` job authenticate to PyPI using GitHub's OIDC token
(`id-token: write`) instead of a long-lived API token secret.

---

## Release checklist

1. Bump the version in `packages/agent-sentinel-sdk/pyproject.toml`.
2. Add a new entry to `packages/agent-sentinel-sdk/CHANGELOG.md` for the
   release.
3. Commit and merge those changes to `main`.
4. Tag the release and push the tag:

   ```bash
   git tag sdk-vX.Y.Z
   git push origin sdk-vX.Y.Z
   ```

5. Watch the `Release SDK` workflow run. It will fail the `build` job if the
   tag version doesn't match `pyproject.toml`, so a mismatch never reaches
   PyPI.

---

## Verify the release

- Check https://pypi.org/project/agentsentinel-sdk/#history for the new
  version.
- Install it in a clean environment: `pip install agentsentinel-sdk==X.Y.Z`.
- Confirm the mirror job pushed the matching source to
  https://github.com/agent-sentinel/agent-sentinel-sdk.

---

## Rollback

PyPI releases are immutable — a bad upload can't be deleted, only
[yanked](https://packaging.python.org/en/latest/guides/using-testpypi/#uploading-and-yanking-releases):

1. Go to https://pypi.org/manage/project/agentsentinel-sdk/release/X.Y.Z/
2. Click "Options" -> "Yank release" and give a reason.

Yanking hides the version from default resolution (existing pins that name it
exactly still work) without removing it. Follow up with a new patch tag that
fixes the issue.

---

## Support

- **Issues**: https://github.com/agent-sentinel/agent-sentinel-sdk/issues
- **Email**: hello@agentsentinel.dev
- **Website**: https://agentsentinel.dev
