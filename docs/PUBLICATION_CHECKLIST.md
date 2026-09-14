# Publication readiness checklist

The source package and three completed branches have artifact-backed provenance.
Before declaring the paper artifact final:

- [ ] Finish the matched no-morphology `1e-5` control to 6,240 steps.
- [ ] Generate its sealed map and, only if the evaluation protocol remains
      unchanged and authorized, run the independent-point evaluation.
- [ ] Update `paper/experiments.yaml` and `paper/results.yaml` from its completion
      markers; never extrapolate the partial trajectory.
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

The current E3m-vs-old-E2 map smoothness comparison is descriptive because both
morphology and DMI learning rate differ. A morphology-effect claim requires the
completed matched control.
