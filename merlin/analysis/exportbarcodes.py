from merlin.core import analysistask


class ExportBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that filters barcodes based on area and mean
    intensity.
    """

    def setup(self) -> None:
        super().setup(parallel=False)

        self.add_dependencies("filter_task")
        self.set_default_parameters(
            {"columns": ["barcode_id", "global_x", "global_y", "cell_index"], "exclude_blanks": True}
        )

        self.define_results(("barcodes", {"index": False}))

        self.columns = self.parameters["columns"]
        self.excludeBlanks = self.parameters["exclude_blanks"]

    def run_analysis(self):
        self.barcodes = self.filter_task.get_barcode_database().get_barcodes(columnList=self.columns)

        if self.excludeBlanks:
            codebook = self.filter_task.get_codebook()
            self.barcodes = self.barcodes[self.barcodes["barcode_id"].isin(codebook.get_coding_indexes())]
