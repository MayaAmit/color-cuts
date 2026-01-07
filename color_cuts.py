#!/usr/bin/env python3
"""
Color Cuts: Foreground/Background Galaxy Separation via Pixel Mask

This script uses color-color space pixel voting to separate foreground
and background galaxies based on a training set with known redshifts.

Usage:
    python color_cuts.py                    # Uses config.yaml in current directory
    python color_cuts.py --config my.yaml   # Uses specified config file
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table, vstack


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_pixel_voting_map(
    color_bg: np.ndarray,
    color_ub: np.ndarray,
    redshift: np.ndarray,
    z_thresh: float,
    xlim: tuple,
    ylim: tuple,
    pixel_size: float,
    purity_threshold: float,
    min_pixel_count: int = 10,
    training_mask: np.ndarray = None,
    weighting: bool = False,
    color_bg_err: np.ndarray = None,
    color_ub_err: np.ndarray = None
) -> tuple:
    """
    Create a pixel-based voting map in color-color space.
    
    Each pixel is classified as foreground or background based on the
    fraction of training objects below/above the redshift threshold.
    
    Parameters
    ----------
    color_bg : array-like
        B-G colors
    color_ub : array-like
        U-B colors
    redshift : array-like
        Redshift values
    z_thresh : float
        Redshift threshold for foreground/background split
    xlim, ylim : tuple
        Color axis limits (min, max)
    pixel_size : float
        Size of each pixel in color space
    purity_threshold : float
        Minimum fraction of foreground objects for pixel to be foreground
    min_pixel_count : int
        Minimum objects in pixel to include in mask
    training_mask : array-like, optional
        Boolean mask for high-quality training data
    weighting : bool
        If True, use variance-based weighting with color errors
    color_bg_err, color_ub_err : array-like, optional
        Color errors (required if weighting=True)
        
    Returns
    -------
    vote_map : 2D array
        1 = background, -1 = foreground, 2 = foreground with high confidence
    x_edges, y_edges : arrays
        Bin edges for the pixel grid
    """
    # Make copies to avoid modifying originals
    color_bg = np.array(color_bg, dtype=float)
    color_ub = np.array(color_ub, dtype=float)
    redshift = np.array(redshift, dtype=float)
    
    if weighting:
        if color_bg_err is None or color_ub_err is None:
            raise ValueError("color_bg_err and color_ub_err required when weighting=True")
        color_bg_err = np.array(color_bg_err, dtype=float)
        color_ub_err = np.array(color_ub_err, dtype=float)
    
    # Apply training mask if provided
    if training_mask is not None:
        color_bg = color_bg[training_mask]
        color_ub = color_ub[training_mask]
        redshift = redshift[training_mask]
        if weighting:
            color_bg_err = color_bg_err[training_mask]
            color_ub_err = color_ub_err[training_mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(color_bg) | np.isnan(color_ub) | np.isnan(redshift))
    if weighting:
        valid_mask &= ~(np.isnan(color_bg_err) | np.isnan(color_ub_err))
    
    color_bg = color_bg[valid_mask]
    color_ub = color_ub[valid_mask]
    redshift = redshift[valid_mask]
    if weighting:
        color_bg_err = color_bg_err[valid_mask]
        color_ub_err = color_ub_err[valid_mask]
    
    # Create redshift masks
    background_mask = redshift > z_thresh
    foreground_mask = redshift <= z_thresh
    
    # Create pixel grid
    x_edges = np.arange(xlim[0], xlim[1] + pixel_size, pixel_size)
    y_edges = np.arange(ylim[0], ylim[1] + pixel_size, pixel_size)
    
    # Calculate actual object counts (unweighted)
    bg_counts, _, _ = np.histogram2d(
        color_bg[background_mask], color_ub[background_mask],
        bins=[x_edges, y_edges]
    )
    fg_counts, _, _ = np.histogram2d(
        color_bg[foreground_mask], color_ub[foreground_mask],
        bins=[x_edges, y_edges]
    )
    actual_counts = bg_counts + fg_counts
    
    # Calculate weights if requested
    if weighting:
        combined_error = (color_bg_err + color_ub_err) / 2.0
        epsilon = 1e-10
        weights = 1.0 / (combined_error**2 + epsilon)
        
        bg_weights = weights[background_mask]
        fg_weights = weights[foreground_mask]
        
        bg_hist, _, _ = np.histogram2d(
            color_bg[background_mask], color_ub[background_mask],
            bins=[x_edges, y_edges], weights=bg_weights
        )
        fg_hist, _, _ = np.histogram2d(
            color_bg[foreground_mask], color_ub[foreground_mask],
            bins=[x_edges, y_edges], weights=fg_weights
        )
    else:
        bg_hist = bg_counts
        fg_hist = fg_counts
    
    # Create voting map based on purity threshold
    vote_map = np.zeros_like(bg_hist)
    total_objects = bg_hist + fg_hist
    has_objects = total_objects > 0
    
    # Calculate purity
    purity = np.zeros_like(bg_hist)
    purity[has_objects] = fg_hist[has_objects] / total_objects[has_objects]
    
    # Apply purity threshold: -1 = foreground, +1 = background
    vote_map[has_objects] = np.where(purity[has_objects] >= purity_threshold, -1, 1)
    
    # Mark high-confidence foreground pixels (>= min_pixel_count objects)
    vote_map_display = vote_map.copy()
    high_confidence_fg = (vote_map == -1) & (actual_counts >= min_pixel_count)
    vote_map_display[high_confidence_fg] = 2  # Green = high confidence foreground
    
    print(f"Pixel map statistics:")
    print(f"  Total pixels with objects: {np.sum(has_objects)}")
    print(f"  Foreground pixels (purity >= {purity_threshold:.0%}): {np.sum(vote_map == -1)}")
    print(f"  High-confidence foreground (>= {min_pixel_count} objects): {np.sum(high_confidence_fg)}")
    print(f"  Background pixels: {np.sum(vote_map == 1)}")
    
    return vote_map_display, x_edges, y_edges, actual_counts


def apply_pixel_mask(
    vote_map: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    catalog: Table,
    cluster_name: str = None,
    use_high_confidence: bool = True,
    redseq_catalog: Table = None
) -> tuple:
    """
    Apply the pixel mask to separate foreground and background galaxies.
    
    Parameters
    ----------
    vote_map : 2D array
        Voting map from create_pixel_voting_map
    x_edges, y_edges : arrays
        Bin edges for the pixel grid
    catalog : Table
        Full catalog to apply mask to
    cluster_name : str, optional
        If provided, filter catalog to this cluster
    use_high_confidence : bool
        If True, only use high-confidence foreground pixels (value=2)
    redseq_catalog : Table, optional
        Red sequence members to add to foreground
        
    Returns
    -------
    foreground_catalog : Table
        Objects classified as foreground
    background_catalog : Table
        Objects classified as background
    """
    # Filter for specific cluster if requested
    if cluster_name is not None and 'CLUSTER' in catalog.colnames:
        cluster_mask = catalog['CLUSTER'] == cluster_name
        working_catalog = catalog[cluster_mask]
        print(f"Filtered to {len(working_catalog)} objects for cluster {cluster_name}")
    else:
        working_catalog = catalog
        print(f"Processing {len(working_catalog)} objects")
    
    # Extract colors
    obj_bg = working_catalog['color_bg'].astype(float)
    obj_ub = working_catalog['color_ub'].astype(float)
    
    # Remove objects with NaN colors
    valid_colors = ~(np.isnan(obj_bg) | np.isnan(obj_ub))
    valid_indices = np.where(valid_colors)[0]
    
    obj_bg = obj_bg[valid_colors]
    obj_ub = obj_ub[valid_colors]
    
    print(f"Objects with valid colors: {len(obj_bg)}")
    
    # Initialize masks
    foreground_mask = np.zeros(len(obj_bg), dtype=bool)
    background_mask = np.zeros(len(obj_bg), dtype=bool)
    
    # Classify each object based on pixel
    for idx, (bg_val, ub_val) in enumerate(zip(obj_bg, obj_ub)):
        # Find pixel indices
        i = np.searchsorted(x_edges, bg_val) - 1
        j = np.searchsorted(y_edges, ub_val) - 1
        
        # Check bounds
        if 0 <= i < len(x_edges)-1 and 0 <= j < len(y_edges)-1:
            pixel_val = vote_map[i, j]
            
            if use_high_confidence:
                # Only high-confidence foreground (value=2)
                if pixel_val == 2:
                    foreground_mask[idx] = True
                elif pixel_val == 1:
                    background_mask[idx] = True
            else:
                # Any foreground pixel (-1 or 2)
                if pixel_val in [-1, 2]:
                    foreground_mask[idx] = True
                elif pixel_val == 1:
                    background_mask[idx] = True
    
    # Apply masks to catalog
    fg_indices = valid_indices[foreground_mask]
    bg_indices = valid_indices[background_mask]
    
    foreground_catalog = working_catalog[fg_indices]
    background_catalog = working_catalog[bg_indices]
    
    # Add red sequence members to foreground if provided
    if redseq_catalog is not None and len(redseq_catalog) > 0:
        # Filter redseq catalog if cluster specified
        if cluster_name is not None and 'CLUSTER' in redseq_catalog.colnames:
            redseq_filtered = redseq_catalog[redseq_catalog['CLUSTER'] == cluster_name]
        else:
            redseq_filtered = redseq_catalog
        
        if len(redseq_filtered) > 0 and 'id' in foreground_catalog.colnames:
            # Remove duplicates
            existing_ids = set(foreground_catalog['id'])
            new_members = redseq_filtered[~np.isin(redseq_filtered['id'], list(existing_ids))]
            if len(new_members) > 0:
                foreground_catalog = vstack([foreground_catalog, new_members])
                print(f"Added {len(new_members)} red sequence members to foreground")
    
    print(f"\nClassification results:")
    print(f"  Foreground: {len(foreground_catalog)} objects")
    print(f"  Background: {len(background_catalog)} objects")
    
    return foreground_catalog, background_catalog


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Separate foreground/background galaxies using color cuts"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Please create a config.yaml file or specify one with --config")
        sys.exit(1)
    
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load catalogs
    print(f"\nLoading redshift catalog: {config['redshift_catalog']}")
    redshift_cat = Table.read(config['redshift_catalog'])
    print(f"  Loaded {len(redshift_cat)} objects with known redshifts")
    
    print(f"\nLoading full catalog: {config['full_catalog']}")
    full_cat = Table.read(config['full_catalog'])
    print(f"  Loaded {len(full_cat)} total objects")
    
    # Load optional red sequence catalog
    redseq_cat = None
    if config.get('redseq_catalog') and config.get('include_redseq'):
        print(f"\nLoading red sequence catalog: {config['redseq_catalog']}")
        redseq_cat = Table.read(config['redseq_catalog'])
        print(f"  Loaded {len(redseq_cat)} red sequence members")
    
    # Extract training data
    redshift = redshift_cat['Z_best'].astype(float)
    color_bg = redshift_cat['color_bg'].astype(float)
    color_ub = redshift_cat['color_ub'].astype(float)
    color_bg_err = redshift_cat['color_bg_err'].astype(float)
    color_ub_err = redshift_cat['color_ub_err'].astype(float)
    
    # Build training mask
    valid_data = ~(np.isnan(color_bg) | np.isnan(color_ub) | np.isnan(redshift))
    good_colors = (color_bg_err < config['error_threshold']) & (color_ub_err < config['error_threshold'])
    
    # Apply redshift source filter if specified
    if config.get('redshift_source_filter'):
        redshift_source = np.array([s.strip() for s in redshift_cat['Z_source']])
        source_filter = redshift_source == config['redshift_source_filter']
        training_mask = valid_data & good_colors & source_filter
        print(f"\nFiltering to {config['redshift_source_filter']} sources")
    else:
        training_mask = valid_data & good_colors
    
    print(f"Training objects after quality cuts: {np.sum(training_mask)}")
    
    # Create pixel voting map
    print("\nCreating pixel voting map...")
    vote_map, x_edges, y_edges, counts = create_pixel_voting_map(
        color_bg=color_bg,
        color_ub=color_ub,
        redshift=redshift,
        z_thresh=config['z_threshold'],
        xlim=tuple(config['xlim']),
        ylim=tuple(config['ylim']),
        pixel_size=config['pixel_size'],
        purity_threshold=config['purity_threshold'],
        min_pixel_count=config['min_pixel_count'],
        training_mask=training_mask,
        weighting=config.get('use_weighting', False),
        color_bg_err=color_bg_err if config.get('use_weighting') else None,
        color_ub_err=color_ub_err if config.get('use_weighting') else None
    )
    
    # Apply mask to full catalog
    print("\nApplying pixel mask to full catalog...")
    foreground, background = apply_pixel_mask(
        vote_map=vote_map,
        x_edges=x_edges,
        y_edges=y_edges,
        catalog=full_cat,
        cluster_name=config.get('cluster_name'),
        use_high_confidence=True,
        redseq_catalog=redseq_cat if config.get('include_redseq') else None
    )
    
    # Save outputs
    cluster = config.get('cluster_name', 'catalog')
    purity = config['purity_threshold']
    pixel = config['pixel_size']
    
    fg_filename = f"{cluster}_foreground_purity{purity}_pixel{pixel}.fits"
    bg_filename = f"{cluster}_background_purity{purity}_pixel{pixel}.fits"
    
    fg_path = output_dir / fg_filename
    bg_path = output_dir / bg_filename
    
    print(f"\nSaving foreground catalog: {fg_path}")
    foreground.write(fg_path, overwrite=True)
    
    print(f"Saving background catalog: {bg_path}")
    background.write(bg_path, overwrite=True)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
