import importlib
import textwrap

import networkx

from merlin.core import analysistask, dataset


def expand_as_string(task: analysistask.ParallelAnalysisTask) -> str:
    """Generate the expand function for the output of a parallel analysis task."""
    filename = task.dataSet.analysis_done_filename(task.get_analysis_name(), "{g}")
    return f"expand('{filename}', g={list(task.fragment_list())})"


def generate_output(
    task: analysistask.AnalysisTask, reftask: analysistask.AnalysisTask = None, *, full_output: bool = False
) -> str:
    """Generate the output string for a task.

    If `reftask` is given, then the output string is constructed to be used for the
    input of `reftask`. If both `task` and `refTask` are parallel tasks with the same
    fragment list, then only the output for a single fragment is needed for the input.
    If `task` is a parellel task but `reftask` is not parallel, or parallel with a
    different fragment list (e.g. Optimize), then the output of all fragments for `task`
    are required for input to `reftask` and the expand string is returned.
    """
    if isinstance(task, analysistask.ParallelAnalysisTask):
        if full_output:
            return expand_as_string(task)
        depends_all = False
        if reftask:
            if isinstance(reftask, analysistask.ParallelAnalysisTask):
                if set(task.fragment_list()) != set(reftask.fragment_list()):
                    depends_all = True
            else:
                depends_all = True
        if not depends_all:
            return f"'{task.dataSet.analysis_done_filename(task, '{i}')}'"
        return expand_as_string(task)
    return f"'{task.dataSet.analysis_done_filename(task)}'"


def generate_input(task: analysistask.AnalysisTask) -> str:
    """Generate the input string for a task."""
    input_tasks = [task.dataSet.load_analysis_task(x) for x in task.get_dependencies()]
    return ",".join([generate_output(x, task) for x in input_tasks]) if len(input_tasks) > 0 else ""


def generate_message(task: analysistask.AnalysisTask) -> str:
    """Generate the message string for a task."""
    message = f"Running {task.get_analysis_name()}"
    if isinstance(task, analysistask.ParallelAnalysisTask):
        message += " {wildcards.i}"
    return message


def generate_shell_command(task: analysistask.AnalysisTask, python_path: str) -> str:
    """Generate the shell command for a task."""
    args = [
        python_path,
        "-m merlin",
        f"-t {task.analysisName}",
        f"-e {task.dataSet.dataHome}",
        f"-s {task.dataSet.analysisHome}",
    ]
    if task.dataSet.profile:
        args.append("--profile")
    if isinstance(task, analysistask.ParallelAnalysisTask):
        args.append("-i {wildcards.i}")
    args.append(task.dataSet.dataSetName)
    return " ".join(args)


def snakemake_rule(task: analysistask.AnalysisTask, python_path: str = "python") -> str:
    """Generate the snakemake rule for a task."""
    string = f"""
    rule {task.get_analysis_name()}:
        input: {generate_input(task)}
        output: {generate_output(task)}
        message: "{generate_message(task)}"
        shell: "{generate_shell_command(task, python_path)}"
    """
    return textwrap.dedent(string).strip()


class SnakefileGenerator:
    def __init__(self, parameters, dataset: dataset.DataSet, python_path: str = None):
        self.parameters = parameters
        self.dataset = dataset
        self.python_path = python_path

    def parse_parameters(self) -> dict[str: analysistask.AnalysisTask]:
        """Create a dict of analysis tasks from the parameters."""
        tasks = {}
        for task_dict in self.parameters["analysis_tasks"]:
            module = importlib.import_module(task_dict["module"])
            analysis_class = getattr(module, task_dict["task"])
            parameters = task_dict.get("parameters")
            name = task_dict.get("analysis_name")
            task = analysis_class(self.dataset, parameters, name)
            if task.get_analysis_name() in tasks:
                raise Exception(
                    "Analysis tasks must have unique names. " + task.get_analysis_name() + " is redundant."
                )
            # TODO This should be more careful to not overwrite an existing
            # analysis task that has already been run.
            task.save()
            tasks[task.get_analysis_name()] = task
        return tasks

    def identify_terminal_tasks(self, tasks: dict[str: analysistask.AnalysisTask]) -> list[str]:
        """Find the terminal tasks."""
        graph = networkx.DiGraph()
        for x in tasks:
            graph.add_node(x)

        for x, a in tasks.items():
            for d in a.get_dependencies():
                graph.add_edge(d, x)

        return [k for k, v in graph.out_degree if v == 0]

    def generate_workflow(self) -> str:
        """Generate a snakemake workflow for the analysis parameters.

        Returns
            the path to the generated snakemake workflow
        """
        tasks = self.parse_parameters()
        terminal_tasks = self.identify_terminal_tasks(tasks)
        terminal_input = ",".join([generate_output(tasks[x], full_output=True) for x in terminal_tasks])
        terminal_rule = f"""
        rule all:
            input: {terminal_input}
        """.strip()
        task_rules = [snakemake_rule(x, self.python_path) for x in tasks.values()]
        snakemake_string = "\n\n".join([textwrap.dedent(terminal_rule).strip()] + task_rules)

        return self.dataset.save_workflow(snakemake_string)
