import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
import rtree
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree

from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
import pandas
import networkx as nx


class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.

    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_data_organization().get_z_positions())
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions()))])


class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_gpickle(
            'cleaned_cells', self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        #fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
        #                           fovBoxes[int(fragmentIndex)].intersects(x)])
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[np.nonzero(allFOVs == fragmentIndex)[0][0]].intersects(x)])
                                   # modified by bereket, previously the code was trying to us teh FOV number as index the list, fovBOX
                                # which is a list of the geometry object for each fov, but this fails if the fov is missing

        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, int(fragmentIndex),
                                               allFOVs, fovBoxes)

        self.dataSet.save_graph_as_gpickle(
            graph, 'cleaned_cells', self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)


# class CellposeSegment(analysistask.AnalysisTask):
#
#     def setup(self) -> None:
#         super().setup(parallel=True)
#
#         self.add_dependencies({"global_align_task": []})
#         self.add_dependencies({"flat_field_task": []}, optional=True)
#
#         self.set_default_parameters(
#             {
#                 "channel": "DAPI",
#                 "z_pos": None,
#                 "diameter": None,
#                 "cellprob_threshold": None,
#                 "flow_threshold": None,
#                 "minimum_size": None,
#                 "dilate_cells": None,
#                 "downscale_xy": 1,
#                 "downscale_z": 1,
#             }
#         )
#
#         self.define_results("mask", ("cell_metadata", {"index": False}))
#
#         self.channelIndex = self.dataSet.get_data_organization().get_data_channel_index(self.parameters["channel"])
#
#     def load_mask(self):
#         mask = self.load_result("mask")
#         if mask.ndim == 3:
#             shape = (
#                 mask.shape[0] * self.parameters["downscale_z"],
#                 mask.shape[1] * self.parameters["downscale_xy"],
#                 mask.shape[2] * self.parameters["downscale_xy"],
#             )
#             z_int = np.round(np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
#             x_int = np.round(np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
#             y_int = np.round(np.linspace(0, mask.shape[2] - 1, shape[2])).astype(int)
#             return mask[z_int][:, x_int][:, :, y_int]
#         shape = (
#             mask.shape[0] * self.parameters["downscale_xy"],
#             mask.shape[1] * self.parameters["downscale_xy"],
#         )
#         x_int = np.round(np.linspace(0, mask.shape[0] - 1, shape[0])).astype(int)
#         y_int = np.round(np.linspace(0, mask.shape[1] - 1, shape[1])).astype(int)
#         return mask[x_int][:, y_int]
#
#     def load_cell_metadata(self):
#         return self.dataSet.load_dataframe_from_csv(
#             "cell_metadata", self.analysis_name, self.fragment, subdirectory="cell_metadata"
#         )
#
#     def load_image(self, zIndex):
#         image = self.dataSet.get_raw_image(self.channelIndex, self.fragment, zIndex)
#         if "flat_field_task" in self.dependencies:
#             image = self.flat_field_task.process_image(image)
#         return image[:: self.parameters["downscale_xy"], :: self.parameters["downscale_xy"]]
#
#     def run_analysis(self):
#         if self.parameters["z_pos"] is not None:
#             zIndex = self.dataSet.position_to_z_index(self.parameters["z_pos"])
#             inputImage = self.load_image(zIndex)
#         else:
#             zPositions = self.dataSet.get_z_positions()[:: self.parameters["downscale_z"]]
#             inputImage = np.array([self.load_image(self.dataSet.position_to_z_index(zIndex)) for zIndex in zPositions])
#         model = cpmodels.Cellpose(gpu=False, model_type="cyto2")
#         if inputImage.ndim == 2:
#             mask, _, _, _ = model.eval(
#                 inputImage,
#                 channels=[0, 0],
#                 diameter=self.parameters["diameter"],
#                 cellprob_threshold=self.parameters["cellprob_threshold"],
#                 flow_threshold=self.parameters["flow_threshold"],
#             )
#         else:
#             frames, _, _, _ = model.eval(list(inputImage))
#             mask = np.array(utils.stitch3D(frames))
#         if self.parameters["minimum_size"]:
#             sizes = pd.DataFrame(regionprops_table(mask, properties=["label", "area"]))
#             mask[np.isin(mask, sizes[sizes["area"] < self.parameters["minimum_size"]]["label"])] = 0
#         if self.parameters["dilate_cells"]:
#             if mask.ndim == 2:
#                 mask = expand_labels(mask, self.parameters["dilate_cells"])
#             else:
#                 mask = np.array([expand_labels(frame, self.parameters["dilate_cells"]) for frame in mask])
#         cell_metadata = pd.DataFrame(regionprops_table(mask, properties=["label", "area", "centroid"]))
#         columns = ["cell_id", "volume"]
#         if mask.ndim == 3:
#             columns.append("z")
#         columns.extend(["x", "y"])
#         cell_metadata.columns = columns
#         downscale = self.parameters["downscale_xy"]
#         cell_metadata["x"] *= downscale
#         cell_metadata["y"] *= downscale
#         if mask.ndim == 3:
#             cell_metadata["z"] *= self.parameters["downscale_z"]
#             cell_metadata["volume"] *= downscale * downscale * self.parameters["downscale_z"]
#         else:
#             cell_metadata["volume"] *= downscale * downscale
#         global_x, global_y = self.global_align_task.fov_coordinates_to_global(
#             self.fragment, cell_metadata[["x", "y"]].T.to_numpy()
#         )
#         cell_metadata["global_x"] = global_x
#         cell_metadata["global_y"] = global_y
#         self.mask = mask
#         self.cell_metadata = cell_metadata
#
#     def metadata(self) -> dict:
#         return {"cells": len(self.cell_metadata)}
