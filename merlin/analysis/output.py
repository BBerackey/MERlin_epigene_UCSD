import numpy as np
import pandas as pd
import scanpy as sc

from merlin.core import analysistask


class FinalOutput(analysistask.AnalysisTask):
    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies({"partition_task": [], "segment_task": [], "link_cell_task": []})

        self.define_results(
            ("cell_metadata", {"index": True}),
            ("detected_transcripts", {"index": False}),
            ("cell_by_gene", {"index": True}),
            "scanpy_object"
        )

    def _combine_overlap_volumes(self):
        volumes = []
        for fragment in self.dataSet.get_overlap_names():
            self.link_cell_task.fragment = fragment
            volumes.append(self.link_cell_task.load_result("overlap_volume"))
        volumes = pd.concat(volumes)
        return volumes.groupby("label").max()

    def get_scanpy_object(self):
        return self.dataSet.load_scanpy_analysis_result("scanpy_object", self)

    def get_cell_metadata_table(self):
        try:
            self.cell_metadata = self.dataSet.load_dataframe_from_csv("cell_metadata", self.analysis_name, index_col=0)
        except FileNotFoundError:
            dfs = []
            cell_mapping = self.link_cell_task.get_cell_mapping()
            for fov in self.dataSet.get_fovs():
                self.segment_task.fragment = fov
                df = self.segment_task.load_metadata()
                df["cell_id"] = fov + "__" + df["cell_id"].astype(str)
                df = df.rename(columns={"volume": "fov_volume"})
                dfs.append(df)
            metadata = pd.concat(dfs).set_index("cell_id")
            metadata["overlap_volume"] = self._combine_overlap_volumes()
            metadata["overlap_volume"] = metadata["overlap_volume"].fillna(0)
            metadata["nonoverlap_volume"] = metadata["fov_volume"] - metadata["overlap_volume"]
            metadata.index = [
                cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in metadata.index
            ]
            metadata.index.name = "cell_id"
            metadata = metadata.groupby("cell_id").agg(
                {"global_x": "mean", "global_y": "mean", "overlap_volume": "mean", "nonoverlap_volume": "sum"}
            )
            metadata["volume"] = metadata["overlap_volume"] + metadata["nonoverlap_volume"]
            metadata = metadata.drop(columns=["overlap_volume", "nonoverlap_volume"])
            self.cell_metadata = metadata

    def run_analysis(self):
        cell_mapping = self.link_cell_task.get_cell_mapping()

        barcodes = self.partition_task.get_barcode_table()
        barcodes["cell_id"] = [
            cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in barcodes["cell_id"]
        ]
        self.detected_transcripts = barcodes

        matrix = self.partition_task.get_cell_by_gene_matrix()
        matrix.index = [cell_mapping[cell_id] if cell_id in cell_mapping else cell_id for cell_id in matrix.index]
        matrix = matrix.reset_index().groupby("index").sum()
        matrix.index.name = None
        self.cell_by_gene = matrix

        self.get_cell_metadata_table()
        blank_cols = np.array(["notarget" in col or "blank" in col.lower() for col in matrix])
        adata = sc.AnnData(matrix.loc[:, ~blank_cols], dtype=np.uint32)
        adata.obsm["X_blanks"] = matrix.loc[:, blank_cols].to_numpy()
        adata.uns["blank_names"] = matrix.columns[blank_cols].to_list()
        adata.obsm["X_spatial"] = np.array(self.cell_metadata[["global_x", "global_y"]].reindex(index=adata.obs.index))
        adata.obs["volume"] = self.cell_metadata["volume"]
        adata.obs["fov"] = [cell_id.split("__")[0] for cell_id in adata.obs.index]
        adata.layers["counts"] = adata.X
        sc.pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata, base=2)
        sc.pp.neighbors(adata, n_neighbors=30, use_rep="X", metric="cosine")
        sc.tl.leiden(adata)
        sc.tl.umap(adata, min_dist=0.3)
        self.scanpy_object = adata
