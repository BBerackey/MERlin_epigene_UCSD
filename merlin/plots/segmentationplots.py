import numpy as np
from matplotlib import pyplot as plt

from merlin.analysis.partition import PartitionBarcodesFromMask
from merlin.analysis.segment import CellposeSegment, FeatureSavingAnalysisTask, LinkCellsInOverlaps
from merlin.plots import tools
from merlin.plots._base import AbstractPlot


class SegmentationBoundaryPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"segment_task": FeatureSavingAnalysisTask})

    def create_plot(self, **kwargs):
        feature_db = kwargs["tasks"]["segment_task"].get_feature_database()
        features = feature_db.read_features()

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", "datalim")

        if len(features) == 0:
            return fig

        z_position = 0
        if len(features[0].get_boundaries()) > 1:
            z_position = int(len(features[0].get_boundaries()) / 2)

        features_z = [feature.get_boundaries()[int(z_position)] for feature in features]
        features_z = [x for y in features_z for x in y]
        coords = [
            [feature.exterior.coords.xy[0].tolist(), feature.exterior.coords.xy[1].tolist()] for feature in features_z
        ]
        coords = [x for y in coords for x in y]
        plt.plot(*coords)

        plt.xlabel("X position (microns)")
        plt.ylabel("Y position (microns)")
        plt.title("Segmentation boundaries")
        return fig


class CellposeBoundaryPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"segment_task": CellposeSegment, "link_cell_task": LinkCellsInOverlaps})
        self.formats = [".png"]

    def plot_mask(self, fov, ax) -> None:
        self.segment_task.fragment = fov
        mask = self.segment_task.load_mask()
        channel = self.dataset.get_data_organization().get_data_channel_index(self.segment_task.parameters["channel"])
        if self.segment_task.parameters["z_pos"] is not None:
            z_index = self.dataset.position_to_z_index(self.segment_task.parameters["z_pos"])
            image = self.dataset.get_raw_image(channel, fov, z_index)
        else:
            z_positions = self.dataset.get_z_positions()
            z_index = z_positions[len(z_positions) // 2]
            image = self.dataset.get_raw_image(channel, fov, z_index)
            mask = mask[int(z_index)]

        ax.imshow(image, cmap="gray")
        ax.contour(
            mask,
            [x + 0.5 for x in np.unique(mask)],
            colors="tab:blue",
            linewidths=1,
            zorder=2,
        )
        ax.contourf(mask, [x + 0.5 for x in np.unique(mask)], colors="tab:blue", alpha=0.2)
        ax.axis("off")

    def create_plot(self, **kwargs) -> plt.Figure:
        self.segment_task = kwargs["tasks"]["segment_task"]
        self.dataset = self.segment_task.dataSet
        link_cell_task = kwargs["tasks"]["link_cell_task"]
        metadata = link_cell_task.load_result("cell_metadata")
        metadata["fov"] = [cell_id.split("__")[0] for cell_id in metadata.index]

        fig, ax = plt.subplots(3, 2, figsize=(8, 12), dpi=300)
        counts = metadata.groupby("fov").count().sort_values("volume")
        fovs = counts.index
        self.plot_mask(fovs[len(fovs) // 2], ax[0, 0])
        ax[0, 0].set_title(f"FOV {fovs[len(fovs)//2]} - {counts.loc[fovs[len(fovs)//2]].volume} cells")
        self.plot_mask(fovs[len(fovs) // 2 - 1], ax[0, 1])
        ax[0, 1].set_title(f"FOV {fovs[len(fovs)//2-1]} - {counts.loc[fovs[len(fovs)//2-1]].volume} cells")
        self.plot_mask(fovs[0], ax[1, 0])
        ax[1, 0].set_title(f"FOV {fovs[0]} - {counts.loc[fovs[0]].volume} cells")
        self.plot_mask(fovs[1], ax[1, 1])
        ax[1, 1].set_title(f"FOV {fovs[1]} - {counts.loc[fovs[1]].volume} cells")
        self.plot_mask(fovs[-1], ax[2, 0])
        ax[2, 0].set_title(f"FOV {fovs[-1]} - {counts.loc[fovs[-1]].volume} cells")
        self.plot_mask(fovs[-2], ax[2, 1])
        ax[2, 1].set_title(f"FOV {fovs[-2]} - {counts.loc[fovs[-2]].volume} cells")

        return fig


class CellVolumeHistogramPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"link_cell_task": LinkCellsInOverlaps})

    def create_plot(self, **kwargs) -> plt.Figure:
        link_cell_task = kwargs["tasks"]["link_cell_task"]
        metadata = link_cell_task.load_result("cell_metadata")
        fig = tools.plot_histogram(metadata, "volume")
        plt.xlabel("Cell volume (pixels)")
        return fig


class BarcodesAssignedToCellsPlot(AbstractPlot):
    def __init__(self, plot_task) -> None:
        super().__init__(plot_task)
        self.set_required_tasks({"partition_task": PartitionBarcodesFromMask, "segment_task": CellposeSegment})
        self.formats = [".png"]

    def create_plot(self, **kwargs) -> plt.Figure:
        partition_task = kwargs["tasks"]["partition_task"]
        segment_task = kwargs["tasks"]["segment_task"]
        partition_task.fragment = self.plot_task.dataSet.get_fovs()[0]
        segment_task.fragment = self.plot_task.dataSet.get_fovs()[0]
        barcodes = partition_task.load_result("barcodes")
        image = segment_task.load_image(zIndex=10)
        incells = barcodes[barcodes["cell_id"] != "000__0"]
        outcells = barcodes[barcodes["cell_id"] == "000__0"]
        fig = plt.figure(dpi=200, figsize=(10, 10))
        plt.imshow(image, cmap="gray", vmax=np.percentile(image, 99))
        plt.scatter(incells["x"], incells["y"], s=1, alpha=0.5, c="tab:blue", marker=".")
        plt.scatter(outcells["x"], outcells["y"], s=1, alpha=0.5, c="tab:red", marker=".")
        plt.axis("off")
        return fig
