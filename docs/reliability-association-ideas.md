# Association and uncertainty follow-up — 9 September 2026

Read-only primary-literature review while the frozen M27 matrix runs. These
are prospective hypotheses, not an M28 declaration, implemented tracker,
qualification result or reason to alter the active experiment. M27 survivors,
if any, take the complete original Stage B first. No experimental mesh is built.

## A different correspondence mechanism

KinectFusion's camera-tracking section uses perspective projection into vertex
and normal images to associate measurements, with compatibility checks, and
parallelizes the linear-system accumulation. Its full system also uses a
volumetric model and reports difficulties when scene motion makes that model
inconsistent with current measurements. Those are properties of that published
system, not measured performance or proof of suitability for this MINI.
[Original Microsoft Research paper, camera-tracking section and dynamic-scene discussion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/kinectfusion-uist-comp.pdf).

Our inference: projective association deserves a distinct preflight because
the existing M27 fitter still chooses Euclidean nearest-point correspondences;
changing directions does not change that association rule. A possible bounded
review would compare camera-ray associations with the existing correspondences
at fixed saved poses, before fitting any new trajectory. It must explicitly
define direction, nearest-pixel rounding, behind-camera/outside-image/zero-depth
refusals, original depth-band behavior, duplicate-pixel ownership, occlusion and
distance gates. Test pixel round trips and rigid-transform direction against
independent arithmetic, and retain every missing match. No inferred geometry,
new object crop, CAD identity, hole filling or volumetric mesh follows from this
idea. Changed association requires its own exact observation controls and full
geometric validation, not inheritance of M27's normal tests.

## Why an inverse Hessian is not a reliability certificate

Brossard, Bonnabel and Barrau explicitly account for initialization uncertainty
and correlated sensor bias as well as measurement noise in ICP covariance.
Their analysis explains why a local linearization alone omits wrong-convergence
uncertainty, and why independent-noise assumptions can be overconfident.
[Author preprint, accepted December 2019, revision January 2020](https://arxiv.org/pdf/1909.05722).

This supports caution, not a new diagnosis of this scanner. Our existing
106,491-step conditioning review already finds overlapping condition ratios
for good and failed trajectories. A future fixed-start perturbation experiment
could describe attraction-basin sensitivity, but without a calibrated pose-prior
and sensor error model it must not label that spread a calibrated uncertainty
or tune a retrospective acceptance threshold. The old forty D same-view fits
already succeed from both prescribed starts; repeating them with many extra
nearby seeds may add little information and is not automatically the next test.

## Retrieval and scope notes

The search also surfaced Stoiber et al., *Iterative Corresponding Geometry*
(CVPR 2022), but direct PDF and HTML retrieval returned internal errors. Its
algorithm is not implemented or relied on from a search excerpt. Publication
dates here come from paper identity, not search-engine crawl-age labels.

The existing C depth render was visually reinspected. It does not independently
identify the printed target, distinguish all support geometry, or establish a
CAD reference; no semantic mask was created. Current algorithm/source hashes,
raw inputs, scanner timing and qualification gates remain unchanged by this
review. The conditional M27 Stage B wrapper is separate preparation, not an
implementation of these literature ideas.
