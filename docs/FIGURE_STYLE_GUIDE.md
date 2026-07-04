# SpatialSelf Figure Style Guide

This project uses the D0/D3/D9/D21 MERFISH story figure as the default visual language for manuscript figures.

## Shared Polish Logic

- Make the background recede before adding more color.
- Give key observations a crisp boundary: thin black outlines for focal points, arrows, boxplot dots, or hotspot rings.
- Use black or near-black axes when the panel is quantitative; use very pale grid lines only when they help reading values.
- Keep annotations outside the data field whenever possible.
- Do not use in-panel explanatory sentences; move them to captions.
- Use one focal color per panel and avoid adding extra hues for secondary information.
- Prefer fewer, clearer marks over many semi-transparent marks.
- Match legends to plotted marks: filled boxes with thin black edges for distributions, bordered colorbars for continuous maps, and no decorative legend frames.
- If a reference-style figure has strong black axes and outlined marks, transfer the boundary logic rather than copying its colors literally.

## Horizontal Bar Charts

- White panel background.
- Low-saturation colors only:
  - focal SpatialSelf/LR item: muted brick red `#C94F48`;
  - supportive biological LR items: muted warm gold `#D6B06A` / `#D7BA7D`;
  - negative or background items: warm gray `#BDB8AE`.
- Horizontal bars with height around `0.58`, no bar edge.
- Sort bars from low to high so the most important/highest values sit near the top.
- Hide top, right, and left spines.
- Keep only the bottom spine, thin and warm gray.
- Use vertical x-axis grid lines only; do not use horizontal row-cutting grid lines.
- Highlight the focal LR by:
  - making the y tick label bold and red;
  - adding a small open downward triangle just past the bar end.
- Use compact sans-serif text:
  - panel title around 7 pt;
  - tick labels around 5.5 pt;
  - x-axis label around 5.4 pt.
- Keep small panel notes inside the lower-right corner, using muted text.

## Spatial Panels

- Use a light warm neutral tissue background.
- Use one dominant signal color per story panel.
- Keep downstream hotspot outlines thin, dark, and low alpha.
- Avoid explanatory text inside the panel unless it is needed to read the data.
- Communication edges should be drawn from source-data sender/receiver coordinates and exported as their own source table.
- When true edges are short at whole-slide scale, increase line width/alpha modestly but do not artificially relocate the edge.

## Boxplots With Overlaid Points

- Use white background and no grid.
- Use black left and bottom spines; hide top and right spines.
- Show all or sampled individual points over each box.
- Points should be filled, high-alpha, and have thin black outlines.
- Use a blue/control group on the left and a red/focal group on the right when comparing background versus SpatialSelf hotspot.
- Use pale box fills matched to the point color.
- Keep medians, whiskers, and caps black.
- Put the p-value above a simple horizontal comparison line.
- Use a compact top legend with small filled boxes when the group colors are not obvious from axis labels alone.
