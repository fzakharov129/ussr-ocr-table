"""Display saved cells over original pages in a notebook, optionally save PNGs."""

import re
from pathlib import Path

from .entities import CellType, CellDirection
from .image_io import list_pages, load_page
from .pipeline import PipelineStage
from .storage import BookStorage, book_output_dir, source_fingerprint
from .validations import ImageChecks, VisualizationChecks, PipelineChecks, CellInspectionChecks


COLORS = {
    CellType.ROW_HEADER: "#00bfff",
    CellType.COLUMN_HEADER: "#ff9900",
    CellType.VALUE: "#00a850",
    CellType.EMPTY: "#cc44ff",
}


def _load_stage_data(book, stage, outputs_dir=None, book_id=None):
    """Shared trusted checkpoint loading for page and cell inspection."""
    checkpoint_stage = VisualizationChecks.checkpoint_stage(stage, PipelineStage)
    zip_path = Path(book).resolve()
    folder = book_output_dir(zip_path, outputs_dir, book_id)
    storage = BookStorage(folder, source_fingerprint(zip_path))
    # A completed snapshot is loaded once; no detection, OCR or external calls.
    data = storage.load(checkpoint_stage)
    PipelineChecks.state(checkpoint_stage, data)
    if checkpoint_stage == PipelineStage.ASSEMBLY:
        graphs_by_page = data.graphs_by_page
        cells_by_page = {name: list(graph.nodes.values())
                         for name, graph in graphs_by_page.items()}
    elif checkpoint_stage == PipelineStage.STRUCTURE:
        graphs_by_page = data
        cells_by_page = {name: list(graph.nodes.values()) for name, graph in data.items()}
    else:
        graphs_by_page = None
        cells_by_page = data
    pages = list_pages(zip_path)
    VisualizationChecks.pages(cells_by_page, pages)

    return zip_path, cells_by_page, graphs_by_page


def _load_stage_cells(book, stage, outputs_dir=None, book_id=None):
    zip_path, cells_by_page, _ = _load_stage_data(book, stage, outputs_dir, book_id)
    return zip_path, cells_by_page


def _draw_graph_edges(ax, graph, selected_ids=None):
    """Draw saved reciprocal edges once, without recomputing the graph."""
    from matplotlib.collections import LineCollection
    groups = {"horizontal": [], "vertical": []}
    seen = set()
    for cell_id in sorted(graph.nodes):
        if selected_ids is not None and cell_id not in selected_ids:
            continue
        for direction, neighbors in graph.neighbors[cell_id].items():
            for neighbor_id in sorted(neighbors):
                if selected_ids is not None and neighbor_id not in selected_ids:
                    continue
                pair = tuple(sorted((cell_id, neighbor_id)))
                if pair in seen:
                    continue
                seen.add(pair)
                group = "horizontal" if direction in (CellDirection.LEFT, CellDirection.RIGHT) else "vertical"
                groups[group].append((graph.nodes[cell_id].center, graph.nodes[neighbor_id].center))
    for name, color in (("horizontal", "#0080ff"), ("vertical", "#ff2020")):
        if groups[name]:
            ax.add_collection(LineCollection(groups[name], colors=color, linewidths=1.6,
                                             alpha=1.0, zorder=2))


def show_book_stage(
    book: str | Path,
    stage: str = "detection",
    *,
    page: str | None = None,
    cell_ids: int | list[int] | None = None,
    outputs_dir: str | Path | None = None,
    book_id: str | None = None,
    show_values: bool = False,
    show_ids: bool = True,
    show_boxes: bool | None = None,
    show_graph: bool = False,
    save_dir: str | Path | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int = 120,
) -> list[Path]:
    """Show every original ZIP page with checkpoint bbox, center and cell ID.

    book is the ZIP path. result (or assembly) selects the final BookResult.
    show_boxes=None hides boxes when showing graphs; True/False overrides this.
    show_values adds recognized_value (falling back to ocr_value) inside bbox.
    show_graph draws saved edges/nodes for structure or result/assembly only.
    save_dir=None writes no images. show=False permits saving without display.
    Returns saved PNG paths, or [] when only displaying in the notebook.
    """
    if show_boxes is None:
        show_boxes = not show_graph
    VisualizationChecks.options(stage, show, save_dir, figsize, dpi, show_values, show_graph, show_ids, show_boxes)
    zip_path, cells_by_page, graphs_by_page = _load_stage_data(book, stage, outputs_dir, book_id)
    CellInspectionChecks.options(page, None, None, True)
    CellInspectionChecks.page_exists(page, cells_by_page)
    selected_ids = CellInspectionChecks.cell_ids(cell_ids)
    cells_by_page = {
        name: [cell for cell in cells
               if selected_ids is None or cell.cell_id in selected_ids]
        for name, cells in cells_by_page.items()
        if page is None or name == page
    }
    if selected_ids is not None:
        cells_by_page = {name: cells for name, cells in cells_by_page.items() if cells}
    pages = list(cells_by_page)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    destination = Path(save_dir) if save_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, page_name in enumerate(pages, start=1):
        image = load_page(zip_path, page_name)
        height, width = image.shape[:2]
        # Validate bbox before drawing, using the existing crop checks.
        for cell in cells_by_page[page_name]:
            ImageChecks.crop(image, cell.bbox)
        size = figsize if figsize is not None else (14, 14 * height / width + 0.5)
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        try:
            ax.imshow(image[:, :, ::-1])  # BGR -> RGB
            if show_graph:
                _draw_graph_edges(ax, graphs_by_page[page_name], selected_ids)
            for cell in cells_by_page[page_name]:
                x, y, w, h = cell.bbox
                color = COLORS[cell.cell_label]
                box = Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1)
                if show_boxes:
                    ax.add_patch(box)
                else:
                    # Keep a transformed rectangle for clipping value text only.
                    box.set_transform(ax.transData)
                cx, cy = cell.center
                ax.plot(cx, cy, marker="o", markersize=6 if show_graph else 3,
                        color=color, markeredgecolor="white" if show_graph else color,
                        markeredgewidth=0.6, zorder=3)
                if show_ids:
                    ax.annotate(str(cell.cell_id), (cx, cy), xytext=(4, 4),
                                textcoords="offset points", fontsize=8, color="black",
                                zorder=4, bbox={"facecolor": "white", "alpha": 0.8,
                                      "edgecolor": "none", "pad": 1})
                if show_values:
                    value = cell.recognized_value
                    if value is None:
                        value = cell.ocr_value
                    if value is not None:
                        # Keep text within the bbox, with ID separately above the center.
                        text = ax.text(cx, y + 0.8 * h, str(value), ha="center", va="center",
                                       fontsize=7, color="black", clip_on=True,
                                       bbox={"facecolor": "white", "alpha": 0.65,
                                             "edgecolor": "none", "pad": 1})
                        text.set_clip_path(box)
            ax.set_title(f"{zip_path.stem} | {stage} | {index}/{len(pages)} | {page_name}")
            if show_graph:
                from matplotlib.lines import Line2D
                ax.legend(handles=[
                    Line2D([0], [0], color="#0080ff", label="left / right"),
                    Line2D([0], [0], color="#ff2020", label="up / down"),
                ], loc="upper right", fontsize=8)
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ax.set_axis_off()
            fig.tight_layout()
            if destination is not None:
                # Index prevents collisions from nested ZIP paths with equal basenames.
                stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(page_name).stem)[:100]
                suffix = "_graph" if show_graph else ""
                path = destination / f"{index:04d}_{stem}_{stage}{suffix}.png"
                fig.savefig(path, dpi=dpi)
                saved.append(path)
            if show:
                plt.show()
        finally:
            plt.close(fig)
    return saved
