import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas
from skimage import transform

from merlin.analysis import decode
from merlin.data.codebook import Codebook
from merlin.util import aberration, decoding, registration


class OptimizeIteration(decode.BarcodeSavingParallelAnalysisTask):

    """
    An analysis task for performing a single iteration of scale factor
    optimization.
    """

    def setup(self) -> None:
        super().setup(parallel=True)

        self.add_dependencies({"preprocess_task": [], "warp_task": ["drifts"]})
        self.add_dependencies({"previous_iteration": []}, optional=True)
        self.set_default_parameters(
            {
                "fov_per_iteration": 50,
                "area_threshold": 5,
                "optimize_background": False,
                "optimize_chromatic_correction": False,
                "crop_width": 0,
                "n_jobs": 1,
            }
        )

        self.define_results(
            "previous_scale_factors",
            "previous_backgrounds",
            "previous_chromatic_corrections",
            "select_frame",
            "scale_factors",
            "background_factors",
            "chromatic_corrections",
            "barcode_counts"
        )

        if "fov_index" in self.parameters:
            logger = self.dataSet.get_logger(self)
            logger.info("Setting fov_per_iteration to length of fov_index")
            self.parameters["fov_per_iteration"] = len(self.parameters["fov_index"])
        else:
            path = self.path / "select_frame"
            files = path.glob("select_frame_*")
            self.parameters["fov_index"] = [file.stem.split("select_frame_")[-1] for file in files]
            if len(self.parameters["fov_index"]) < self.parameters["fov_per_iteration"]:
                zIndices = list(range(len(self.dataSet.get_z_positions())))
                combinations = set(itertools.product(self.dataSet.get_fovs(), zIndices))
                combinations -= {tuple(zslice.split("__")) for zslice in self.parameters["fov_index"]}
                for zslice in np.random.choice(
                    [f"{fovIndex}__{zIndex}" for fovIndex, zIndex in combinations],
                    size=self.parameters["fov_per_iteration"] - len(self.parameters["fov_index"]),
                    replace=False,
                ):
                    self.parameters["fov_index"].append(zslice)
        self.fragment_list = self.parameters["fov_index"]

    def get_codebook(self) -> Codebook:
        return self.preprocess_task.get_codebook()

    def run_analysis(self):
        codebook = self.get_codebook()

        fovIndex, zIndex = self.fragment.split("__")
        zIndex = int(zIndex)

        scaleFactors = self._get_previous_scale_factors()
        backgrounds = self._get_previous_backgrounds()
        chromaticTransformations = self._get_previous_chromatic_transformations()

        self.previous_scale_factors = scaleFactors
        self.save_result("previous_scale_factors")
        self.previous_backgrounds = backgrounds
        self.save_result("previous_backgrounds")
        self.previous_chromatic_corrections = chromaticTransformations
        self.save_result("previous_chromatic_corrections")

        self.select_frame = np.array([fovIndex, zIndex])
        self.save_result("select_frame")

        chromaticCorrector = aberration.RigidChromaticCorrector(chromaticTransformations, self.get_reference_color())
        self.chromatic_corrections = chromaticTransformations
        preprocess_task = self.dataSet.load_analysis_task("DeconvolutionPreprocessGuo", fovIndex)
        warpedImages = preprocess_task.get_processed_image_set(
            fovIndex, zIndex=zIndex, chromaticCorrector=chromaticCorrector
        )

        decoder = decoding.PixelBasedDecoder(codebook)
        areaThreshold = self.parameters["area_threshold"]
        decoder.refactorAreaThreshold = areaThreshold
        di, pm, npt, d = decoder.decode_pixels(
            warpedImages, scaleFactors, backgrounds, n_jobs=self.parameters["n_jobs"]
        )

        self.scale_factors, self.background_factors, self.barcode_counts = decoder.extract_refactors(
            di, pm, npt, extractBackgrounds=self.parameters["optimize_background"]
        )

        # TODO this saves the barcodes under fragment instead of fov
        # the barcodedb should be made more general
        cropWidth = self.parameters["crop_width"]
        self.get_barcode_database().write_barcodes(
            pandas.concat(
                [
                    decoder.extract_barcodes_with_index(
                        i, di, pm, npt, d, fovIndex, cropWidth, zIndex, minimumArea=areaThreshold
                    )
                    for i in range(codebook.get_barcode_count())
                ]
            ),
            fov=self.fragment,
        )

    def _get_used_colors(self) -> List[str]:
        dataOrganization = self.dataSet.get_data_organization()
        codebook = self.get_codebook()
        return sorted(
            {
                dataOrganization.get_data_channel_color(dataOrganization.get_data_channel_for_bit(x))
                for x in codebook.get_bit_names()
            }
        )

    def _calculate_initial_scale_factors(self) -> np.ndarray:
        bitCount = self.get_codebook().get_bit_count()

        initialScaleFactors = np.zeros(bitCount)
        pixelHistograms = self.preprocess_task.get_pixel_histogram()
        for i in range(bitCount):
            cumulativeHistogram = np.cumsum(pixelHistograms[i])
            cumulativeHistogram = cumulativeHistogram / cumulativeHistogram[-1]
            # Add two to match matlab code.
            # TODO: Does +2 make sense? Used to be consistent with Matlab code
            initialScaleFactors[i] = np.argmin(np.abs(cumulativeHistogram - 0.9)) + 2

        return initialScaleFactors

    def _get_previous_scale_factors(self) -> np.ndarray:
        if "previous_iteration" not in self.parameters:
            scaleFactors = self._calculate_initial_scale_factors()
        else:
            previousIteration = self.dataSet.load_analysis_task(self.parameters["previous_iteration"], "")
            scaleFactors = previousIteration.get_scale_factors()

        return scaleFactors

    def _get_previous_backgrounds(self) -> np.ndarray:
        if "previous_iteration" not in self.parameters:
            backgrounds = np.zeros(self.get_codebook().get_bit_count())
        else:
            previousIteration = self.dataSet.load_analysis_task(self.parameters["previous_iteration"], "")
            backgrounds = previousIteration.get_backgrounds()

        return backgrounds

    def _get_previous_chromatic_transformations(self) -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        if "previous_iteration" not in self.parameters:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform() for v in usedColors if v >= u} for u in usedColors}
        else:
            previousIteration = self.dataSet.load_analysis_task(self.parameters["previous_iteration"], "")
            return previousIteration._get_chromatic_transformations()

    # TODO the next two functions could be in a utility class. Make a
    #  chromatic aberration utility class

    def get_reference_color(self):
        return min(self._get_used_colors())

    def get_chromatic_corrector(self) -> aberration.ChromaticCorrector:
        """Get the chromatic corrector estimated from this optimization
        iteration

        Returns:
            The chromatic corrector.
        """
        return aberration.RigidChromaticCorrector(self._get_chromatic_transformations(), self.get_reference_color())

    def _get_chromatic_transformations(self) -> Dict[str, Dict[str, transform.SimilarityTransform]]:
        """Get the estimated chromatic corrections from this optimization
        iteration.

        Returns:
            a dictionary of dictionary of transformations for transforming
            the farther red colors to the most blue color. The transformation
            for transforming the farther red color, e.g. '750', to the
            farther blue color, e.g. '560', is found at result['560']['750']
        """
        #if not self.is_complete():
        #    raise Exception("Analysis is still running. Unable to get scale " + "factors.")

        if not self.parameters["optimize_chromatic_correction"]:
            usedColors = self._get_used_colors()
            return {u: {v: transform.SimilarityTransform() for v in usedColors if v >= u} for u in usedColors}

        try:
            return self.dataSet.load_pickle_analysis_result("chromatic_corrections", self.analysis_name)
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError, EOFError):
            # TODO - this is messy. It can be broken into smaller subunits and
            # most parts could be included in a chromatic aberration class
            previousTransformations = self._get_previous_chromatic_transformations()
            previousCorrector = aberration.RigidChromaticCorrector(previousTransformations, self.get_reference_color())
            codebook = self.get_codebook()
            dataOrganization = self.dataSet.get_data_organization()

            barcodes = self.get_barcode_database().get_barcodes()
            uniqueFOVs = np.unique(barcodes["fov"])

            usedColors = self._get_used_colors()
            colorPairDisplacements = {u: {v: [] for v in usedColors if v >= u} for u in usedColors}

            for fov in uniqueFOVs:

                fovBarcodes = barcodes[barcodes["fov"] == fov]
                zIndexes = np.unique(fovBarcodes["z"])
                for z in zIndexes:
                    currentBarcodes = fovBarcodes[fovBarcodes["z"] == z]
                    # TODO this can be moved to the run function for the task
                    # so not as much repeated work is done when it is called
                    # from many different tasks in parallel
                    warpedImages = np.array(
                        [
                            self.dataSet.load_analysis_task(self.parameters["warp_task"], fov).get_aligned_image(
                                fov, dataOrganization.get_data_channel_for_bit(b), int(z), previousCorrector
                            )
                            for b in codebook.get_bit_names()
                        ]
                    )

                    for i, cBC in currentBarcodes.iterrows():
                        onBits = np.where(codebook.get_barcode(cBC["barcode_id"]))[0]

                        # TODO this can be done by crop width when decoding
                        if (
                            cBC["x"] > 10
                            and cBC["y"] > 10
                            and warpedImages.shape[1] - cBC["x"] > 10
                            and warpedImages.shape[2] - cBC["y"] > 10
                        ):

                            refinedPositions = np.array(
                                [
                                    registration.refine_position(warpedImages[i, :, :], cBC["x"], cBC["y"])
                                    for i in onBits
                                ]
                            )
                            for p in itertools.combinations(enumerate(onBits), 2):
                                c1 = dataOrganization.get_data_channel_color(p[0][1])
                                c2 = dataOrganization.get_data_channel_color(p[1][1])

                                if c1 < c2:
                                    colorPairDisplacements[c1][c2].append(
                                        [
                                            np.array([cBC["x"], cBC["y"]]),
                                            refinedPositions[p[1][0]] - refinedPositions[p[0][0]],
                                        ]
                                    )
                                else:
                                    colorPairDisplacements[c2][c1].append(
                                        [
                                            np.array([cBC["x"], cBC["y"]]),
                                            refinedPositions[p[0][0]] - refinedPositions[p[1][0]],
                                        ]
                                    )

            tForms = {}
            for k, v in colorPairDisplacements.items():
                tForms[k] = {}
                for k2, v2 in v.items():
                    tForm = transform.SimilarityTransform()
                    goodIndexes = [i for i, x in enumerate(v2) if not any(np.isnan(x[1])) and not any(np.isinf(x[1]))]
                    tForm.estimate(
                        np.array([v2[i][0] for i in goodIndexes]), np.array([v2[i][0] + v2[i][1] for i in goodIndexes])
                    )
                    tForms[k][k2] = tForm + previousTransformations[k][k2]

            self.chromatic_corrections = tForms
            self.save_result("chromatic_corrections")

            return tForms

    def get_scale_factors(self) -> np.ndarray:
        """Get the final, optimized scale factors.

        Returns:
            a one-dimensional numpy array where the i'th entry is the
            scale factor corresponding to the i'th bit.
        """
        #if not self.is_complete():
        #    raise Exception("Analysis is still running. Unable to get scale " + "factors.")

        try:
            return self.load_result("scale_factors")
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError, EOFError, IndexError):
            refactors = np.array(
                [
                    self.dataSet.load_analysis_task(self.analysis_name, i).load_result("scale_factors")
                    for i in self.fragment_list
                ]
            )

            # Don't rescale bits that were never seen
            refactors[refactors == 0] = 1

            previousFactors = np.array(
                [
                    self.dataSet.load_analysis_task(self.analysis_name, i).load_result(
                        "previous_scale_factors"
                    )
                    for i in self.fragment_list
                ]
            )

            scaleFactors = np.nanmedian(np.multiply(refactors, previousFactors), axis=0)

            self.scale_factors = scaleFactors
            self.save_result("scale_factors")

            return scaleFactors

    def get_backgrounds(self) -> np.ndarray:
        #if not self.is_complete():
        #    raise Exception("Analysis is still running. Unable to get " + "backgrounds.")

        try:
            return self.load_result("backgrounds")
        # OSError and ValueError are raised if the previous file is not
        # completely written
        except (FileNotFoundError, OSError, ValueError, EOFError, IndexError):
            refactors = np.array(
                [
                    self.dataSet.load_analysis_task(self.analysis_name, i).load_result("background_factors")
                    for i in self.fragment_list
                ]
            )

            previousBackgrounds = np.array(
                [
                    self.dataSet.load_analysis_task(self.analysis_name, i).load_result("previous_backgrounds")
                    for i in self.fragment_list
                ]
            )

            previousFactors = np.array(
                [
                    self.dataSet.load_analysis_task(self.analysis_name, i).load_result(
                        "previous_scale_factors"
                    )
                    for i in self.fragment_list
                ]
            )

            backgrounds = np.nanmedian(np.add(previousBackgrounds, np.multiply(refactors, previousFactors)), axis=0)

            self.backgrounds = backgrounds
            self.save_result("backgrounds")

            return backgrounds

    def get_scale_factor_history(self) -> np.ndarray:
        """Get the scale factors cached for each iteration of the optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            scale factor corresponding to the i'th bit in the j'th
            iteration.
        """
        if "previous_iteration" not in self.parameters:
            return np.array([self.get_scale_factors()])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters["previous_iteration"], ""
            ).get_scale_factor_history()
            return np.append(previousHistory, [self.get_scale_factors()], axis=0)

    def get_barcode_count_history(self) -> np.ndarray:
        """Get the set of barcode counts for each iteration of the
        optimization.

        Returns:
            a two-dimensional numpy array where the i,j'th entry is the
            barcode count corresponding to the i'th barcode in the j'th
            iteration.
        """
        countsMean = np.mean(
            [
                self.dataSet.load_analysis_task(self.analysis_name, i).load_result("barcode_counts")
                for i in self.fragment_list
            ],
            axis=0,
        )

        if "previous_iteration" not in self.parameters:
            return np.array([countsMean])
        else:
            previousHistory = self.dataSet.load_analysis_task(
                self.parameters["previous_iteration"], ""
            ).get_barcode_count_history()
            return np.append(previousHistory, [countsMean], axis=0)
