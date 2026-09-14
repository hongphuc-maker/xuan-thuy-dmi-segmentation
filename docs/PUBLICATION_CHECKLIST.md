# Publication readiness checklist

The source package and four completed branches have artifact-backed provenance.
Before declaring the paper artifact final:

- [x] Finish the matched no-morphology `1e-5` control to 6,240 steps.
- [x] Generate its sealed map and, with the unchanged authorized protocol, run
      the independent-point evaluation.
- [x] Update `paper/experiments.yaml` and `paper/results.yaml` from its completion
      markers; do not extrapolate from the earlier partial trajectory.
- [ ] Confirm article title, author order, affiliations, citation text, and
      copyright holder in `CITATION.cff` and `LICENSE`.
- [ ] Add the final data-availability statement and a stable data/artifact archive
      identifier if redistribution is authorized.
- [ ] Replace provisional software citation metadata with the article DOI when
      available.
- [ ] Run the complete test suite and `scripts/audit_public_tree.py` from a clean
      clone.
- [ ] Create a signed or annotated release tag and archive that exact release in
      a long-term repository such as Zenodo.

The matched no-morphology LR `1e-5` control is now complete, so metric comparisons
between it and the LR `1e-5` morphology branch are no longer confounded by the
DMI learning rate. The existing map-smoothness numbers in `paper/results.yaml`
still compare the morphology branch against the historical LR `3e-7`
no-morphology branch. Do not reinterpret those topology numbers as the matched
morphology effect until a direct E3m-versus-control map-comparison artifact is
generated.
