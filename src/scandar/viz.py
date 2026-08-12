"""Figures.  *(brief, "Visualization of Results")*

One shared matplotlib style — consistent palette, 200 dpi, readable in print — so
every figure in the report looks like it came from the same project.

The set the report is built around:

* dataset contact sheets, and the degradation pipeline as a step-by-step strip
* synthetic samples with corners overlaid, plus the alignment-verification
  difference map that proves input and target line up
* the "spot the fake" panel: generated samples beside real photos. If a stranger
  can instantly tell which is which, the degradations are not realistic enough
* training curves, and the loss ablation as zoomed text crops where MSE's blur is
  visible next to the combined loss's strokes
* real-photo triplets: rectified input, our output, reference scan
* corner predictions over ground truth, per-corner error boxplots, PCK curves,
  predicted heatmaps, and a gallery of failures
* the dropout study's synthetic-versus-real gap, before and after
* the end-to-end storyboard: photo -> predicted corners -> rectified -> enhanced
"""
