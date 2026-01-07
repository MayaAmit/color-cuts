# Color Cuts

Foreground/background galaxy separation using color-color pixel mask classification.

This tool uses a training catalog with known redshifts to build a pixel-based voting map in color-color space, then applies that map to classify galaxies in a full photometric catalog as foreground or background.

## Quick Start

### 1. Clone the repo

```bash
cd /projects/mccleary_group/your_username/
git clone https://github.com/YOUR_USERNAME/color-cuts.git
cd color-cuts
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or if you're using conda:
```bash
conda install numpy astropy pyyaml
```

### 3. Configure your paths

Edit `config.yaml` to point to your data:

```yaml
# Training catalog with known redshifts
redshift_catalog: "/path/to/your/redshift_catalog.fits"

# Full catalog to apply the mask to
full_catalog: "/path/to/your/full_catalog.fits"

# Where to save output
output_dir: "./output"
cluster_name: "Abell3411"
```

### 4. Run

```bash
python color_cuts.py
```

Or with a custom config:
```bash
python color_cuts.py --config my_config.yaml
```

## Output

The script produces two FITS files in your `output_dir`:

- `{cluster}_foreground_purity{X}_pixel{Y}.fits` - galaxies classified as foreground
- `{cluster}_background_purity{X}_pixel{Y}.fits` - galaxies classified as background

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `z_threshold` | Redshift cutoff for foreground/background | 0.2 |
| `purity_threshold` | Min fraction of foreground objects to classify pixel as foreground | 0.8 |
| `pixel_size` | Size of pixels in color-color space | 0.05 |
| `min_pixel_count` | Min objects in pixel to be considered high-confidence | 10 |
| `error_threshold` | Max color error for training objects | 20 |
| `xlim` | B-G color range | [-1, 3] |
| `ylim` | U-B color range | [-2, 3.5] |
| `use_weighting` | Use error-weighted purity calculation | false |
| `include_redseq` | Add red sequence members to foreground | false |
| `redshift_source_filter` | Filter training data by source (e.g., "NED") | "NED" |

## Required Catalog Columns

### Redshift catalog (training data)
- `Z_best` - best redshift estimate
- `Z_source` - source of redshift (e.g., "NED", "DESI")
- `color_bg` - B-G color
- `color_ub` - U-B color
- `color_bg_err` - B-G color error
- `color_ub_err` - U-B color error

### Full catalog
- `color_bg` - B-G color
- `color_ub` - U-B color
- `CLUSTER` - cluster name (optional, for filtering)

## How It Works

1. **Build training mask**: Filter redshift catalog for reliable sources and low color errors
2. **Create pixel map**: Bin training data in color-color space, calculate foreground purity per pixel
3. **Classify pixels**: Pixels with purity >= threshold become "foreground", others "background"
4. **Apply to full catalog**: Each galaxy is classified based on which pixel it falls into

## Example

```python
# You can also use the functions directly in Python
from color_cuts import create_pixel_voting_map, apply_pixel_mask
from astropy.table import Table

# Load your data
redshift_cat = Table.read("redshift_catalog.fits")
full_cat = Table.read("full_catalog.fits")

# Create the pixel map
vote_map, x_edges, y_edges, counts = create_pixel_voting_map(
    color_bg=redshift_cat['color_bg'],
    color_ub=redshift_cat['color_ub'],
    redshift=redshift_cat['Z_best'],
    z_thresh=0.2,
    xlim=(-1, 3),
    ylim=(-2, 3.5),
    pixel_size=0.05,
    purity_threshold=0.8
)

# Apply to full catalog
foreground, background = apply_pixel_mask(
    vote_map, x_edges, y_edges,
    catalog=full_cat,
    cluster_name="Abell3411"
)
```
