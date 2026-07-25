# Tensor pipe and semantics

```text
6 × BGR camera planes
  → scalar camera stem / view projection
  → 8 × 8 × 16 current BEV
  → integer ego-translation warp of previous BEV
  → fixed cross-sample spatial attention analogue
  ├→ stable top-k tracks → 3 × 4-step multimodal motion
  ├→ four map-query lines
  ├→ 3 × 8 × 8 future occupancy
  └→ command-conditioned 6-step ego plan → collision score
```

Camera payloads use `camera, channel, y, x` contiguous layout. BEV uses
`y, x, channel`. Coordinates in canonical results are meters in the current ego
frame; positive x is right and positive y is forward. A context carries temporal
state only while scene identity matches. `ua_context_reset` clears it.

The tiny warp rounds ego translation to BEV cells. Its yaw field is validated
but not applied; this is recorded as a candidate gap rather than hidden.
