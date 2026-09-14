# Security policy

## Supported version

Security fixes are applied to the latest release on `main`.

## Private reporting

Use GitHub's private vulnerability-reporting feature when it is enabled. If it
is unavailable, contact the repository owner privately through their GitHub
profile. Do not publish credentials, private Drive links, restricted geospatial
records, or proof-of-concept data exfiltration in an issue.

## Secrets and research data

The project never requires a token embedded in source or a notebook. A public
clone uses no GitHub credential. Store cloud credentials in the platform's
secret manager and pass data/run locations at runtime.

The `.gitignore` excludes common raster, checkpoint, archive, and run outputs,
but contributors must still inspect every staged file before committing.
