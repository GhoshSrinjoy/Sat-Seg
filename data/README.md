# Imagery and labels

- `raw/`: original imagery and labels; preserve provider metadata and acquisition dates.
- `interim/`: tiles, temporary conversions and intermediate artifacts.
- `processed/`: prepared model inputs and reviewed labels.

Large files are ignored by Git. The exploration notebook's Berkeley demonstration
image is cached in `raw/examples/`; a training dataset still needs to be selected.
For geospatial work, use GeoTIFFs with a known CRS and transform. Select RGB
bands explicitly; satellite band orders vary. Split train/evaluation data by
geographic region and date to reduce leakage between neighboring tiles.
