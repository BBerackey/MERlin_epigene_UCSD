import os
import errno
import json
import shutil
import pandas
import numpy as np
import sqlalchemy
import fnmatch
import tifffile
import importlib
import time
import logging
from matplotlib import pyplot as plt
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict

from storm_analysis.sa_library import datareader
import merlin
from merlin.core import analysistask
from merlin.data import dataorganization
from merlin.data import codebook


TaskOrName = Union[analysistask.AnalysisTask, str]


class DataSet(object):

    def __init__(self, dataDirectoryName: str,
                 dataHome: str=None, analysisHome: str=None):
        """Create a dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
        """
        if dataHome is None:
            dataHome = merlin.DATA_HOME
        if analysisHome is None:
            analysisHome = merlin.ANALYSIS_HOME

        self.rawDataPath = os.sep.join([dataHome, dataDirectoryName])
        if not os.path.isdir(self.rawDataPath):
            raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.rawDataPath)
            
        self.analysisPath = os.sep.join([analysisHome, dataDirectoryName])
        os.makedirs(self.analysisPath, exist_ok=True)

        self.logPath = os.sep.join([self.analysisPath, 'logs'])
        os.makedirs(self.logPath, exist_ok=True)

        self.figurePath = os.sep.join([self.analysisPath, 'figures'])
        os.makedirs(self.figurePath, exist_ok=True)

    def save_figure(self, analysisTask: TaskOrName, figure: plt.Figure,
                    figureName: str) -> None:
        """Save the figure into the analysis results for this DataSet

        This function will save the figure in both png and pdf formats.

        Args:
            analysisTask: the analysis task that generated this figure.
            figure: the figure handle for the figure to save
            figureName: the name of the file to store the figure in, excluding
                    extension
        """
        savePath = os.sep.join(
                [self.get_analysis_subdirectory(analysisTask, 'figures'),
                    figureName])

        figure.savefig(savePath + '.png', pad_inches=0)
        figure.savefig(savePath + '.pdf', transparent=True, pad_inches=0)

    def get_analysis_image_set(
            self, analysisTask: TaskOrName, imageBaseName: str,
            imageIndex: int) -> np.ndarray:
        """Get an analysis image set saved in the analysis for this data set.

        Args:
            analysisTask: the analysis task that generated and stored the
                image set.
            imageBaseName: the base name of the image
            imageIndex: index of the image set to retrieve
        """
        return tifffile.imread(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex))

    def get_analysis_image(
            self, analysisTask: TaskOrName, imageBaseName: str, imageIndex: int,
            imagesPerSlice: int, sliceIndex: int,
            frameIndex: int) -> np.ndarray:
        """Get an image from an image set save in the analysis for this
        data set.

        Args:
            analysisTask: the analysis task that generated and stored the
                image set.
            imageBaseName: the base name of the image
            imageIndex: index of the image set to retrieve
            imagesPerSlice: the number of images in each slice of the image
                file
            sliceIndex: the index of the slice to get the image
            frameIndex: the index of the frame in the specified slice
        """
        # TODO - It may be useful to add a function that gets all
        # frames in a slice
        imageFile = tifffile.TiffFile(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex))
        indexInFile = sliceIndex*imagesPerSlice + frameIndex
        return imageFile.asarray(key=int(indexInFile))
    
    def writer_for_analysis_images(
            self, analysisTask: TaskOrName, imageBaseName: str,
            imageIndex: int=None, imagej: bool=True) -> tifffile.TiffWriter:
        """Get a writer for writing tiff files from an analysis task.

        Args:
            analysisTask:
            imageBaseName:
            imageIndex:
            imagej:
        Returns:

        """
        return tifffile.TiffWriter(self._analysis_image_name(
            analysisTask, imageBaseName, imageIndex), imagej=imagej)

    @staticmethod
    def analysis_tiff_description(sliceCount: int, frameCount: int) -> Dict:
        imageDescription = {'ImageJ': '1.47a\n',
                            'images': sliceCount*frameCount,
                            'channels': 1,
                            'slices': sliceCount,
                            'frames': frameCount,
                            'hyperstack': True,
                            'loop': False}
        return imageDescription

    def _analysis_image_name(self, analysisTask: TaskOrName,
                             imageBaseName: str, imageIndex: int) -> str:
        destPath = self.get_analysis_subdirectory(
                analysisTask, subdirectory='images')
        if imageIndex is None:
            return os.sep.join([destPath, imageBaseName+'.tif'])
        else:
            return os.sep.join([destPath, imageBaseName+str(imageIndex)+'.tif'])

    def _analysis_result_save_path(
            self, resultName: str, analysisName: str, resultIndex: int=None,
            subdirectory: str=None) -> str:
        saveName = resultName
        if resultIndex is not None:
            saveName += '_' + str(resultIndex)
        return os.sep.join([self.get_analysis_subdirectory(
            analysisName, subdirectory), saveName])

    def save_dataframe_to_csv(
            self, dataframe: pandas.DataFrame, resultName: str,
            analysisTask: TaskOrName=None, **kwargs) -> None:
        if analysisTask is not None:
            savePath = self._analysis_result_save_path(
                    resultName, analysisTask.get_analysis_name()) + '.csv'
        else:
            savePath = os.sep.join([self.analysisPath, resultName]) + '.csv'

        with open(savePath, 'w') as f:
            dataframe.to_csv(f, **kwargs)

    def load_dataframe_from_csv(
            self, resultName: str, analysisTask: TaskOrName=None,
            **kwargs) -> pandas.DataFrame:
        if analysisTask is not None:
            savePath = self._analysis_result_save_path(
                    resultName, analysisTask.get_analysis_name()) + '.csv'
        else:
            savePath = os.sep.join([self.analysisPath, resultName]) + '.csv'

        with open(savePath, 'r') as f:
            return pandas.read_csv(f, **kwargs)

    def save_analysis_result(
            self, analysisResult: np.ndarray, resultName: str,
            analysisName: str, resultIndex: int=None,
            subdirectory: str=None) -> None:
        # TODO - only implemented currently for ndarray
        if not isinstance(analysisResult, np.ndarray):
            raise TypeError('analysisResult must be a numpy array')

        savePath = self._analysis_result_save_path(
                resultName, analysisName, resultIndex, subdirectory)
        np.save(savePath, analysisResult)
    
    def load_analysis_result(
            self, resultName: str, analysisName: str, resultIndex: int=None,
            subdirectory: str=None) -> np.array:
        # TODO - This should determine the file extension based on the
        # files that are present
        savePath = self._analysis_result_save_path(
                resultName, analysisName, resultIndex, subdirectory) + '.npy'
        return np.load(savePath)

    def get_analysis_subdirectory(
            self, analysisTask: TaskOrName, subdirectory: str=None,
            create: bool=True) -> str:
        """
        analysisTask can either be the class or a string containing the
        class name.

        create - Flag indicating if the analysis subdirectory should be
            created if it does not already exist.
        """
        if isinstance(analysisTask, analysistask.AnalysisTask):
            analysisName = analysisTask.get_analysis_name()
        else:
            analysisName = analysisTask

        if subdirectory is None:
            subdirectoryPath = os.sep.join(
                    [self.analysisPath, analysisName])
        else:
            subdirectoryPath = os.sep.join(
                    [self.analysisPath, analysisName, subdirectory])

        if create:
            os.makedirs(subdirectoryPath, exist_ok=True)

        return subdirectoryPath

    def get_task_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(
                analysisTask, subdirectory='tasks')

    def get_log_subdirectory(self, analysisTask: TaskOrName):
        return self.get_analysis_subdirectory(
                analysisTask, subdirectory='log')
        
    def save_analysis_task(self, analysisTask: analysistask.AnalysisTask):
        saveName = os.sep.join([self.get_task_subdirectory(
            analysisTask), 'task.json'])
        
        with open(saveName, 'w') as outFile:
            json.dump(analysisTask.get_parameters(), outFile, indent=4) 

    def load_analysis_task(self, analysisTaskName: str) \
            -> analysistask.AnalysisTask:
        loadName = os.sep.join([self.get_task_subdirectory(
            analysisTaskName), 'task.json'])

        with open(loadName, 'r') as inFile:
            parameters = json.load(inFile)
            analysisModule = importlib.import_module(parameters['module'])
            analysisTask = getattr(analysisModule, parameters['class'])
            return analysisTask(self, parameters, analysisTaskName)
            
    def delete_analysis(self, analysisTask: TaskOrName) -> None:
        """
        Remove all files associated with the provided analysis 
        from this data set.

        Before deleting an analysis task, it must be verified that the
        analysis task is not running.
        """
        analysisDirectory = self.get_analysis_subdirectory(analysisTask)
        shutil.rmtree(analysisDirectory)

    def analysis_exists(self, analysisTaskName: str) -> bool:
        """
        Determine if an analysis task with the specified name exists in this 
        dataset.
        """
        analysisPath = self.get_analysis_subdirectory(
                analysisTaskName, create=False)
        return os.path.exists(analysisPath)

    def get_logger(self, analysisTask: analysistask.AnalysisTask,
                   fragmentIndex: int=None) -> logging.Logger:
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += '.' + str(fragmentIndex)

        logger = logging.getLogger(loggerName)
        logger.setLevel(logging.DEBUG)
        fileHandler = logging.FileHandler(
                self._log_path(analysisTask, fragmentIndex))
        fileHandler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        return logger

    def close_logger(self, analysisTask: analysistask.AnalysisTask,
                     fragmentIndex=None) -> None:
        loggerName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            loggerName += '.' + str(fragmentIndex)

        logger = logging.getLogger(loggerName)

        handlerList = list(logger.handlers)
        for handler in handlerList:
            logger.removeHandler(handler)
            handler.flush()
            handler.close()

    def _log_path(self, analysisTask: analysistask.AnalysisTask,
                  fragmentIndex=None) -> str:
        logName = analysisTask.get_analysis_name()
        if fragmentIndex is not None:
            logName += '_' + str(fragmentIndex)
        logName += '.log'

        return os.sep.join([self.get_log_subdirectory(analysisTask), logName])

    def _analysis_status_file(self, analysisTask: analysistask.AnalysisTask,
                              eventName: str, fragmentIndex: int=None) -> str:
        if fragmentIndex is None:
            fileName = analysisTask.get_analysis_name() + '.' + eventName
        else:
            fileName = analysisTask.get_analysis_name() + \
                    '_' + str(fragmentIndex) + '.' + eventName
        return os.sep.join([self.get_task_subdirectory(analysisTask),
                fileName])

    def record_analysis_started(self, analysisTask: analysistask.AnalysisTask,
                                fragmentIndex: int=None) -> None:
        self._record_analysis_event(analysisTask, 'start', fragmentIndex)

    def record_analysis_running(self, analysisTask: analysistask.AnalysisTask,
                                fragmentIndex: int=None) -> None:
        self._record_analysis_event(analysisTask, 'run', fragmentIndex)

    def record_analysis_complete(self, analysisTask: analysistask.AnalysisTask,
                                 fragmentIndex: int=None) -> None:
        self._record_analysis_event(analysisTask, 'done', fragmentIndex)

    def record_analysis_error(self, analysisTask: analysistask.AnalysisTask,
                              fragmentIndex: int=None) -> None:
        self._record_analysis_event(analysisTask, 'error', fragmentIndex)

    def _record_analysis_event(
            self, analysisTask: analysistask.AnalysisTask, eventName: str,
            fragmentIndex: int=None) -> None:
        fileName = self._analysis_status_file(
                analysisTask, eventName, fragmentIndex)
        with open(fileName, 'w') as f:
            f.write('%s' % time.time())

    def is_analysis_idle(self, analysisTask: analysistask.AnalysisTask,
                         fragmentIndex: int=None) -> bool:
        fileName = self._analysis_status_file(
                analysisTask, 'run', fragmentIndex)
        return time.time() - os.path.getmtime(fileName) > 120

    def check_analysis_started(self, analysisTask: analysistask.AnalysisTask,
                               fragmentIndex: int=None) -> bool:
        return self._check_analysis_event(analysisTask, 'start', fragmentIndex)

    def check_analysis_running(self, analysisTask: analysistask.AnalysisTask,
                               fragmentIndex: int=None) -> bool:
        return self._check_analysis_event(analysisTask, 'run', fragmentIndex)

    def check_analysis_done(self, analysisTask: analysistask.AnalysisTask,
                            fragmentIndex: int=None) -> bool:
        return self._check_analysis_event(analysisTask, 'done', fragmentIndex)

    def check_analysis_error(self, analysisTask: analysistask.AnalysisTask,
                             fragmentIndex: int=None) -> bool:
        return self._check_analysis_event(analysisTask, 'error', fragmentIndex)

    def _check_analysis_event(
            self, analysisTask: analysistask.AnalysisTask, eventName: str,
            fragmentIndex: int=None) -> bool:
        fileName = self._analysis_status_file(
                analysisTask, eventName, fragmentIndex)
        return os.path.exists(fileName)

    def get_database_engine(self, analysisTask: analysistask.AnalysisTask=None,
                            index: int=None):
        if analysisTask is None:
            return sqlalchemy.create_engine('sqlite:///' +
                    os.sep.join([self.analysisPath, 'analysis_data.db']))
        else:
            dbPath = os.sep.join(
                        [self.analysisPath, analysisTask.get_analysis_name(),
                         'db'])
            os.makedirs(dbPath, exist_ok=True)

            if index is None:
                dbName = 'analysis_data.db'
            else:
                dbName = 'analysis_data' + str(index) + '.db'

            return sqlalchemy.create_engine(os.sep.join(
                        ['sqlite:///' + dbPath, dbName]))


class ImageDataSet(DataSet):

    def __init__(self, dataDirectoryName: str, dataHome: str=None,
                analysisHome: str=None, microscopeParametersName: str=None):
        """Create a dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """
        super().__init__(dataDirectoryName, dataHome, analysisHome)

        if microscopeParametersName is not None:
            self._import_microscope_parameters(microscopeParametersName)
    
        self._load_microscope_parameters()


    def get_image_file_names(self):
        return sorted(
                [os.sep.join([self.rawDataPath, currentFile])
                    for currentFile in os.listdir(self.rawDataPath)
                if currentFile.endswith('.dax') \
                or currentFile.endswith('.tif') \
                or currentFile.endswith('.tiff')])

    def load_image(self, imagePath, frameIndex):
        with datareader.inferReader(imagePath) as reader:
            imageIn = reader.loadAFrame(int(frameIndex))
            if self.transpose:
                imageIn = np.transpose(imageIn)
            if self.flipHorizontal:
                imageIn = np.flip(imageIn, axis=1)
            if self.flipVertical:
                imageIn = np.flip(imageIn, axis=0)
            return imageIn 

    def image_stack_size(self, imagePath):
        """
        Get the size of the image stack stored in the specified image path.

        Returns:
            a three element list with [width, height, frameCount] or None
                    if the file does not exist
        """
        if not os.path.exists(imagePath):
            return None

        with datareader.inferReader(imagePath) as reader:
            return reader.filmSize()

    def _import_microscope_parameters(self, microscopeParametersName):
        sourcePath = os.sep.join([merlin.MICROSCOPE_PARAMETERS_HOME,
                microscopeParametersName])
        destPath = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])

        shutil.copyfile(sourcePath, destPath) 

    def _load_microscope_parameters(self): 
        path = os.sep.join(
                [self.analysisPath, 'microscope_parameters.json'])
        
        if os.path.exists(path):
            with open(path) as inputFile:
                self.microscopeParameters = json.load(inputFile)
        else:
            self.microscopeParameters = {}

        self.flipHorizontal = self.microscopeParameters.get(
            'flip_horizontal', True)
        self.flipVertical = self.microscopeParameters.get(
            'flip_vertical', False)
        self.transpose = self.microscopeParameters.get('transpose', True)
        self.micronsPerPixel = self.microscopeParameters.get(
                'microns_per_pixel', 0.106)
        self.imageDimensions = self.microscopeParameters.get(
                'image_dimensions', [2048, 2048])

    def get_microns_per_pixel(self):
        """Get the conversion factor to convert pixels to microns."""

        return self.micronsPerPixel

    def get_image_dimensions(self):
        """Get the dimensions of the images in this data set.

        Returns:
            A tuple containing the width and height of each image in pixels.
        """
        return self.imageDimensions


class MERFISHDataSet(ImageDataSet):

    def __init__(self, dataDirectoryName: str, codebookName: str=None,
                dataOrganizationName: str=None, positionFileName: str=None,
                dataHome: str=None, analysisHome: str=None,
                microscopeParametersName: str=None):
        """Create a MERFISH dataset for the specified raw data.

        Args:
            dataDirectoryName: the relative directory to the raw data
            codebookName: the name of the codebook to use. The codebook
                    should be present in the analysis parameters
                    directory. A full path can be provided for a codebook
                    present in another directory.
            dataOrganizationName: the name of the data organization to use.
                    The data organization should be present in the analysis
                    parameters directory. A full path can be provided for
                    a codebook present in another directory.
            positionFileName: the name of the position file to use.
            dataHome: the base path to the data. The data is expected
                    to be in dataHome/dataDirectoryName. If dataHome
                    is not specified, DATA_HOME is read from the
                    .env file.
            analysisHome: the base path for storing analysis results. Analysis
                    results for this DataSet will be stored in
                    analysisHome/dataDirectoryName. If analysisHome is not
                    specified, ANALYSIS_HOME is read from the .env file.
            microscopeParametersName: the name of the microscope parameters
                    file that specifies properties of the microscope used
                    to acquire the images represented by this ImageDataSet
        """

        super().__init__(dataDirectoryName, dataHome, analysisHome,
                microscopeParametersName)

        # TODO: it is possible to also extract positions from the images. This
        # should be implemented
        if positionFileName is not None:
            self._import_positions(positionFileName)
        self._load_positions()

        self.dataOrganization = dataorganization.DataOrganization(
                self, dataOrganizationName)
        self.codebook = codebook.Codebook(self, codebookName)

    def get_codebook(self) -> codebook.Codebook:
        return self.codebook

    def get_data_organization(self) -> dataorganization.DataOrganization:
        return self.dataOrganization

    def get_stage_positions(self) -> List[List[float]]:
        return self.positions

    def get_fov_offset(self, fov: int) -> Tuple[float, float]:
        """Get the offset of the specified fov in the global coordinate system.
        This offset is based on the anticipated stage position.

        Args:
            fov: index of the field of view
        Returns:
            A tuple specificing the x and y offset of the top right corner 
            of the specified fov in pixels.
        """
        #TODO - this should be implemented using the position of the fov. 
        return self.positions.loc[fov]['X'], self.positions.loc[fov]['Y']


    def z_index_to_position(self, zIndex: int) -> float:
        """Get the z position associated with the provided z index."""

        return self.get_z_positions()[zIndex]

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this dataset.

        Returns:
            A sorted list of all unique z positions
        """
        return self.dataOrganization.get_z_positions()

    def get_fovs(self) -> List[int]:
        return self.dataOrganization.get_fovs()

    def get_imaging_rounds(self) -> List[int]:
        # TODO - check this function
        return np.unique(self.dataOrganization.fileMap['imagingRound'])

    def get_raw_image(self, dataChannel, fov, zPosition):
        return self.load_image(
                self.dataOrganization.get_image_filename(dataChannel, fov),
                self.dataOrganization.get_image_frame_index(
                    dataChannel, zPosition))

    def get_fiducial_image(self, dataChannel, fov):
        return self.load_image(
                self.dataOrganization.get_fiducial_filename(dataChannel, fov),
                self.dataOrganization.get_fiducial_frame_index(dataChannel))

    def _load_positions(self):
        positionPath = os.sep.join([self.analysisPath, 'positions.csv'])
        #TODO - this is messy searching for the position file
        #TODO - I should check to make sure the number of positions 
        # matches the number of FOVs
        if not os.path.exists(positionPath):
            for f in os.listdir(self.rawDataPath):
                if fnmatch.fnmatch(f, '*position*'):
                    shutil.copyfile(
                            os.sep.join([self.rawDataPath, f]), positionPath)
        
        if not os.path.exists(positionPath):
            for f in os.listdir(os.sep.join([self.rawDataPath, '..'])):
                if fnmatch.fnmatch(f, '*position*'):
                    shutil.copyfile(
                            os.sep.join([self.rawDataPath, '..', f]), 
                            positionPath)
        self.positions = pandas.read_csv(positionPath, header=None,
                names=['X','Y'])

    def _import_positions(self, positionFileName):
        sourcePath = os.sep.join([merlin.POSITION_HOME, \
                positionFileName + '.csv'])
        destPath = os.sep.join([self.analysisPath, 'positions.csv'])
            
        shutil.copyfile(sourcePath, destPath)    

    def _convert_parameter_list(self, listIn, castFunction, delimiter=';'):
        return [castFunction(x) for x in listIn.split(delimiter) if len(x)>0]



