import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Match the styling used in draw plot.py for visual consistency
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica'],
})

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "151673_metadata.csv"
OUTPUT_DIR = BASE_DIR / "cluster_snapshots_less_blurred"
OUTPUT_DIR.mkdir(exist_ok=True)

GT_CANDIDATES = ["layer_guess_reordered", "ManualAnnotation"]

PLOT_CONFIGS = [
    {
        "title": "Ground Truth",
        "source": "ground_truth",
    },
    {
        "title": "CiCLoDS + BayesSpace",
        "path": BASE_DIR / "CICLODS_151673_allocation.csv",
        "cluster_col": "spatial.cluster",
        "id_col": "spot",
        "row_field": "array_row",
        "col_field": "array_col",
        "source": "file_coords",
    },
    {
        "title": "CiCLoDS (Python)",
        "path": BASE_DIR / "151673_ciclods_cluster_allocations.txt",
        "cluster_col": "ciclods_cluster",
        "id_col": "cell_id",
        "source": "metadata",
        "sep": "\t",
    },
    {
        "title": "BayesSpace",
        "path": BASE_DIR / "BayesSpace_151673_allocation.csv",
        "cluster_col": "spatial.cluster",
        "id_col": "spot",
        "row_field": "array_row",
        "col_field": "array_col",
        "source": "file_coords",
    },
    {
        "title": "PCA Baseline",
        "path": BASE_DIR / "151673_pca_cluster_allocations.txt",
        "cluster_col": "matched_baseline_cluster",
        "id_col": "cell_id",
        "source": "metadata",
        "sep": "\t",
    },
    {
        "title": "Full Allocation",
        "path": BASE_DIR / "151673_full_allocation.csv",
        "cluster_col": "cluster",
        "id_col": "spot_id",
        "row_field": "x",
        "col_field": "y",
        "source": "file_coords",
    },
]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata (indexed) and standardize spatial coordinate columns."""
    meta = pd.read_csv(metadata_path)
    index_col = 'barcode' if 'barcode' in meta.columns else meta.columns[0]
    meta = meta.set_index(index_col)
    if 'row' not in meta.columns or 'col' not in meta.columns:
        raise ValueError("Metadata file must include 'row' and 'col' columns.")
    meta['row_coord'] = pd.to_numeric(meta['row'], errors='coerce')
    meta['col_coord'] = pd.to_numeric(meta['col'], errors='coerce')
    gt_col = next((col for col in GT_CANDIDATES if col in meta.columns), None)
    if gt_col is None:
        raise ValueError("No ground-truth annotation column found in metadata.")
    meta['gt_label'] = meta[gt_col].astype(str)
    return meta


def load_allocation(cfg: dict, metadata: pd.DataFrame) -> pd.DataFrame:
    """Return standardized coordinates + cluster labels for a given dataset."""
    read_kwargs = {}
    if cfg.get('sep'):
        read_kwargs['sep'] = cfg['sep']
    if cfg['source'] == 'ground_truth':
        df = metadata[['gt_label', 'row_coord', 'col_coord']].copy()
        df = df.rename(columns={'gt_label': 'cluster_label'})
    else:
        df = pd.read_csv(cfg['path'], **read_kwargs)

    if cfg['source'] == 'metadata':
        df = df.merge(metadata[['row_coord', 'col_coord']], left_on=cfg['id_col'], right_index=True, how='left')
    elif cfg['source'] == 'file_coords':
        df = df.rename(columns={
            cfg['row_field']: 'row_coord',
            cfg['col_field']: 'col_coord',
        })

    if cfg['source'] != 'ground_truth':
        df['cluster_label'] = df[cfg['cluster_col']].astype(str)

    df['row_coord'] = pd.to_numeric(df['row_coord'], errors='coerce')
    df['col_coord'] = pd.to_numeric(df['col_coord'], errors='coerce')
    df = df.dropna(subset=['row_coord', 'col_coord'])
    return df[['cluster_label', 'row_coord', 'col_coord']]


def sort_labels(labels):
    """Sort labels numerically when possible."""

    def sort_key(label):
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)

    return sorted(labels, key=sort_key)


def get_palette(n_labels: int):
    """Fetch up to n_labels colors using a Tab20 palette loop."""
    cmap = plt.get_cmap('tab20')
    return [cmap(i % cmap.N) for i in range(n_labels)]


def order_labels_by_depth(df: pd.DataFrame, label_col: str):
    medians = df.groupby(label_col)['row_coord'].median().sort_values()
    return medians.index.tolist()


def assign_colors_by_depth(df: pd.DataFrame, ordered_colors):
    ordered_labels = order_labels_by_depth(df, 'cluster_label')
    if not ordered_labels:
        ordered_labels = sort_labels(pd.unique(df['cluster_label']))
    color_map = {}
    for idx, label in enumerate(ordered_labels):
        color_map[label] = ordered_colors[idx % len(ordered_colors)]
    # Include any missing labels
    for label in sort_labels(pd.unique(df['cluster_label'])):
        if label not in color_map:
            color_map[label] = ordered_colors[len(color_map) % len(ordered_colors)]
            ordered_labels.append(label)
    return ordered_labels, color_map


def save_individual_plots(panels, ordered_colors, gt_palette, gt_order, output_dir: Path):
    """Save each panel as a separate image file without legends, titles, or axes."""

    for title, df, is_ground_truth in panels:
        # Create a clean filename
        safe_name = title.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").lower()
        output_path = output_dir / f"{safe_name}_less_blurred.png"

        # Setup Figure
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

        # Determine colors
        if is_ground_truth:
            labels = [label for label in gt_order if label in df['cluster_label'].unique()]
            extras = [label for label in sort_labels(pd.unique(df['cluster_label'])) if label not in labels]
            labels.extend(extras)
            color_map = {label: gt_palette.get(label, ordered_colors[i % len(ordered_colors)]) for i, label in
                         enumerate(labels)}
        else:
            labels, color_map = assign_colors_by_depth(df, ordered_colors)

        # Plot points with adjusted smoothing parameters
        for label in labels:
            mask = df['cluster_label'] == label
            ax.scatter(
                df.loc[mask, 'col_coord'],
                -df.loc[mask, 'row_coord'],
                # --- CHANGES FOR LESS BLUR ---
                s=45,  # Decreased size (was 180)
                alpha=0.7,  # Increased opacity (was 0.3)
                # -----------------------------
                c=[color_map[label]],
                edgecolors='none',
            )

        # Remove all decorations
        ax.axis('off')  # Removes spines, ticks, labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0)

        # Save without padding to ensure just the content is saved
        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved less blurred image: {output_path}")


def main():
    # Ensure metadata exists before running
    if not METADATA_PATH.exists():
        print(f"Error: Metadata file not found at {METADATA_PATH}")
        return

    metadata = load_metadata(METADATA_PATH)
    gt_order = order_labels_by_depth(metadata, 'gt_label')
    if not gt_order:
        gt_order = sort_labels(metadata['gt_label'].dropna().unique())
    ordered_colors = get_palette(max(len(gt_order), 1))
    gt_palette = {label: ordered_colors[idx % len(ordered_colors)] for idx, label in enumerate(gt_order)}

    panels = []
    for cfg in PLOT_CONFIGS:
        try:
            # Basic check if file exists for non-metadata sources
            if cfg['source'] != 'ground_truth' and cfg['source'] != 'metadata' and not cfg['path'].exists():
                print(f"Skipping {cfg['title']}: File not found at {cfg['path']}")
                continue

            df = load_allocation(cfg, metadata)
            panels.append((cfg['title'], df, cfg['source'] == 'ground_truth'))
        except Exception as e:
            print(f"Skipping {cfg['title']} due to error: {e}")

    if not panels:
        print("No plots could be generated.")
        return

    save_individual_plots(panels, ordered_colors, gt_palette, gt_order, OUTPUT_DIR)


if __name__ == "__main__":
    main()