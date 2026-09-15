# Publication readiness checklist

The source package and four completed branches have artifact-backed provenance.
Before declaring the paper artifact final:

- [x] Finish the matched no-morphology `1e-5` control to 6,240 steps.
- [x] Generate its sealed map and, with the unchanged authorized protocol, run
      the independent-point evaluation.
- [x] Update `paper/experiments.yaml` and `paper/results.yaml` from its completion
      markers; do not extrapolate from the earlier partial trajectory.
- [x] Run the matched map-fragmentation and paired independent-point comparison.
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

The E3m-vs-old-E2 comparison remains confounded because both morphology and DMI
learning rate differ. The completed E3m-vs-matched-E2w comparison isolates the
configured morphology factor for this seed and scene; it does not establish
multi-seed generality, accuracy superiority, or non-inferiority.
