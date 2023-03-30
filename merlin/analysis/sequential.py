import pandas
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import imagefilters


class SumSignal(analysistask.AnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring
    RNA species that were stained individually.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies("warp_task", "segment_task", "global_align_task")
        self.set_default_parameters({"apply_highpass": False, "highpass_sigma": 5, "z_index": 0})

        self.define_results("sequential_signal")

        if self.parameters["z_index"] >= len(self.dataSet.get_z_positions()):
            raise analysistask.InvalidParameterError(
                "Invalid z_index specified for %s. (%i > %i)"
                % (self.analysis_name, self.parameters["z_index"], len(self.dataSet.get_z_positions()))
            )

        self.highpass = str(self.parameters["apply_highpass"]).upper() == "TRUE"

    def _extract_signal(self, cells, inputImage, zIndex) -> pandas.DataFrame:
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zIndex]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions:
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                    pixels.append(np.array(self.global_align_task.global_coordinates_to_fov(cell.get_fov(), xyZip)))
                cellCoords.append(pixels)

        cellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))]
        mask = np.zeros(inputImage.shape, np.uint8)
        for i, cell in enumerate(cellCoords):
            cv2.drawContours(mask, cell, -1, i + 1, -1)
        propsDict = {x.label: x for x in regionprops(mask, inputImage)}
        propsOut = pandas.DataFrame(
            data=[
                (propsDict[k].intensity_image.sum(), propsDict[k].filled_area) if k in propsDict else (0, 0)
                for k in range(1, len(cellCoords) + 1)
            ],
            index=cellIDs,
            columns=["Intensity", "Pixels"],
        )
        return propsOut

    def _get_sum_signal(self, fov, channels, zIndex):
        cells = self.segment_task.get_feature_database().read_features(fov)

        signals = []
        for ch in channels:
            img = self.warp_task.get_aligned_image(fov, ch, zIndex)
            if self.highpass:
                highPassSigma = self.parameters["highpass_sigma"]
                highPassFilterSize = int(2 * np.ceil(3 * highPassSigma) + 1)
                img = imagefilters.high_pass_filter(img, highPassFilterSize, highPassSigma)
            signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [0]])

        # adding num of pixels
        signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [1]])

        compiledSignal = pandas.concat(signals, 1)
        compiledSignal.columns = channels + ["Pixels"]

        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat([self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()])

        return self.dataSet.load_dataframe_from_csv(
            "sequential_signal", self.analysis_name, fov, "signals", index_col=0
        )

    def run_analysis(self, fragmentIndex):
        zIndex = int(self.parameters["z_index"])
        channels, geneNames = self.dataSet.get_data_organization().get_sequential_rounds()

        fovSignal = self._get_sum_signal(fragmentIndex, channels, zIndex)
        normSignal = fovSignal.iloc[:, :-1].div(fovSignal.loc[:, "Pixels"], 0)
        normSignal.columns = geneNames
        self.sequential_signal = normSignal


class ExportSumSignals(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies("sequential_task")

        self.define_results("sequential_sum_signals")

    def run_analysis(self):
        self.sequential_sum_signals = self.sequential_task.get_sum_signals()
