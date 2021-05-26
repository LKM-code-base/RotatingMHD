import argparse
import subprocess
from enum import Enum
from enum import auto
import pandas as pd
import numpy as np
import os
from os import path


class ApplicationNotExistingError(Exception):
    """Raised when application does not exist"""
    pass


class ExecutingApplicationFailed(Exception):
    """Raised when executing the test failed"""
    pass


class ConvergenceTableExtractionError(Exception):
    """Raised when executing the test failed"""
    pass


class TestType(Enum):
    spatial = auto()
    temporal = auto()
    both = auto()


class DimensionlessNumber(Enum):
    Reynolds = auto()
    Peclet = auto()


class ConvergenceTest:
    _prm_suffix = ".prm"
    _source_suffix = ".cc"
    _prm_string_patterns = ["subsection Dimensionless numbers",
                            "Convergence test type",
                            "Number of spatial convergence cycles",
                            "Number of temporal convergence cycles",
                            "Number of initial global refinements",
                            "Number of global refinements",
                            "Initial time step",
                            "Graphical output directory",
                            "Graphical output frequency"]
    _table_header_str = "===============================================" +\
                        "==============================================="

    def __init__(self, application_name, test_type=TestType.temporal,
                 n_spatial_cycles=2, n_temporal_cycles=2, n_initial_global_refinements=5,
                 n_global_refinements=8, initial_time_step=1e-1, final_time_step=1e-3):

        assert isinstance(application_name, str)
        self._application_name = application_name

        assert isinstance(test_type, TestType)
        self._test_type = test_type

        if self._test_type is TestType.spatial:
            self._test_type_str = "spatial"
        elif self._test_type is TestType.temporal:
            self._test_type_str = "temporal"

        assert isinstance(n_spatial_cycles, int)
        assert n_spatial_cycles > 1
        self._n_spatial_cycles = n_spatial_cycles

        assert isinstance(n_temporal_cycles, int)
        assert n_temporal_cycles > 1
        self._n_temporal_cycles = n_temporal_cycles

        assert isinstance(n_initial_global_refinements, int)
        assert n_initial_global_refinements > 1
        self._n_initial_global_refinements = n_initial_global_refinements

        assert isinstance(n_global_refinements, int)
        assert n_global_refinements > 1
        assert n_global_refinements >= self._n_initial_global_refinements \
                                     + self._n_spatial_cycles
        self._n_global_refinements = n_global_refinements

        assert isinstance(initial_time_step, float)
        assert initial_time_step > 0.0
        self._initial_time_step = initial_time_step

        assert isinstance(final_time_step, float)
        assert final_time_step > 0.0
        assert final_time_step < self._initial_time_step
        self._final_time_step = final_time_step

        string_value_map = dict()
        string_value_map["Convergence test type"] = self._test_type_str
        string_value_map["Number of spatial convergence cycles"] = self._n_spatial_cycles
        string_value_map["Number of temporal convergence cycles"] = self._n_temporal_cycles
        string_value_map["Number of initial global refinements"] = self._n_initial_global_refinements
        string_value_map["Number of global refinements"] = self._n_global_refinements
        string_value_map["Initial time step"] = self._initial_time_step
        string_value_map["Time step"] = self._final_time_step
        self._prm_string_value_map = string_value_map

        self._prm_exclusion_list = ["subsection", "scheme", "#"]

    def _check_convergence_table(self, correct_table, other_table):
        assert isinstance(correct_table, pd.DataFrame)
        assert isinstance(other_table, pd.DataFrame)
        assert len(correct_table) == len(other_table)
        assert hasattr(self, "_exactly_equal_columns")
        assert hasattr(self, "_closely_equal_columns")

        mandatory_columns = self._exactly_equal_columns.union(self._closely_equal_columns)
        for col in mandatory_columns:
            assert col in correct_table
            assert col in other_table

        for col in self._exactly_equal_columns:
            assert all(correct_table[col] == other_table[col])

        matching_error_norms = True
        for col in self._closely_equal_columns:
            atol = 1.0e-12
            rtol = 1.0e-9
            if not np.allclose(other_table[col], correct_table[col], atol=atol, rtol=rtol):
                matching_error_norms = False
                break

        return matching_error_norms

    def _extract_convergence_table(self, input_str):
        assert isinstance(input_str, str)
        assert self._table_header_str not in input_str

        # create list
        table_list = input_str.split(sep="\n")
        table_list = list(filter(lambda x: x != "" and
                                 "velocity" not in x.lower() and
                                 "pressure" not in x.lower(), table_list))
        for i, line in enumerate(table_list):
            line = line.split()
            if "-" in line:
                line = [x if x != "-" else "" for x in line]
            table_list[i] = line

        # extract header
        header = table_list.pop(0)

        # make sure that remaining table is numeric
        for line in table_list:
            for col in line:
                if col.isalpha():
                    raise ConvergenceTableExtractionError()

        # format columns
        columns = []
        for col in header:
            key_found = False
            for key in ("L2", "H1", "Linfty"):
                if key.lower() in col.lower():
                    columns.append((key + "_error").lower())
                    columns.append((key + "_rate").lower())
                    key_found = True
                    break
            if not key_found:
                columns.append(col.lower())

        # convert to numerical type
        def _string_to_numerical_type(x):
            assert isinstance(x, str)
            if "." in x:
                return float(x)
            elif "e" in x:
                return float(x)
            elif x != "":
                return int(x)
            else:
                return None

        table_list = [[_string_to_numerical_type(entry) for entry in row] for row in table_list]

        # create pandas data frame
        for line in table_list:
            assert len(line) == len(columns)
        df = pd.concat([pd.DataFrame([x], columns=columns) for x in table_list], ignore_index=True)
        return df

    def _read_convergence_table_string(self, output):
        # remove output previous to convergence table
        assert self._table_header_str in output
        output = output[output.index(self._table_header_str):]
        output = output.lstrip("=")
        # remove timer ouput
        table_header_str = "+---------------"
        assert table_header_str in output
        output = output[:output.index(table_header_str)]
        assert "+" not in output

        return output

    def _setup_parameter_file(self, fname, modified_fname):
        assert path.exists(fname)
        assert path.isfile(fname)

        modified_lines = []
        with open(fname, "r") as file:
            for line in file:
                assert isinstance(line, str)
                if not any(key in line for key in self._prm_exclusion_list):
                    for key, value in self._prm_string_value_map.items():
                        assert isinstance(key, str)

                        if key in line and "subsection" not in line:
                            assert " = " in line, "The string ' = ' is present " +\
                                                  "in the string: " + line
                            line = line[:line.index(" = ") + 3]
                            line += "{0}".format(value)
                            break
                modified_lines.append(line)

        with open(modified_fname, "w") as modified_file:
            modified_file.write("\n".join(modified_lines))
            modified_file.write("\n")

    def _setup_parameter_mapping(self):
        raise NotImplementedError("You are calling a purely virtual method.")

    def _process_output(self, output):
        raise NotImplementedError("You are calling a purely virtual method.")

    @property
    def name(self):
        return self._application_name

    def run(self, n_processes=1, reference_result=False):
        # current directory
        cwd = os.getcwd()
        # directory checks
        application_dir = path.join(cwd, "applications")
        prm_fname = path.join(application_dir,
                              self._application_name + self._prm_suffix)
        exec_fname = path.join(application_dir, self._application_name)
        if not path.exists(application_dir) or not path.exists(exec_fname)\
            or not path.exists(prm_fname):
            raise ApplicationNotExistingError()
        # create dump directory
        dump_dir = path.join(cwd, "dump")
        if not path.exists(dump_dir):
            os.mkdir(dump_dir)
        # setup problem specific parameters
        self._setup_parameter_mapping()
        # setup parameter
        modified_prm_fname = path.join(dump_dir,
                                       self._application_name + self._prm_suffix)
        self._setup_parameter_file(prm_fname, modified_prm_fname)
        assert path.exists(modified_prm_fname)
        # create symbolic link
        os.symlink(exec_fname, path.join(dump_dir, self._application_name))
        # perform convergence test
        os.chdir(dump_dir)
        print("Running convergence test " + self._application_name + "...",
              end="", flush=True)
        output = subprocess.check_output(["mpirun", "-np", str(n_processes),
                                          "./" + self._application_name], text=True)
        print("   done!")
        # finalize
        os.chdir(cwd)
        subprocess.run(["rm", "-rf", dump_dir], check=True)
        # compare convergence table
        success = self._process_output(output, reference_result)
        assert isinstance(success, bool)
        return success


class ThermalTGVConvergenceTest(ConvergenceTest):
    def __init__(self, peclet_number, test_type=TestType.temporal,
                 n_spatial_cycles=2, n_temporal_cycles=2, n_initial_global_refinements=3,
                 n_global_refinements=3, initial_time_step=1e-1, final_time_step=1e-3):
        super().__init__(application_name="ThermalTGV",
                         test_type=test_type,
                         n_spatial_cycles=n_spatial_cycles,
                         n_temporal_cycles=n_temporal_cycles,
                         n_initial_global_refinements=n_initial_global_refinements,
                         n_global_refinements=n_global_refinements,
                         initial_time_step=initial_time_step,
                         final_time_step=final_time_step)

        assert isinstance(peclet_number, float)
        assert peclet_number > 0.0
        self._peclet_number = peclet_number

    def _setup_parameter_mapping(self):
        self._prm_string_value_map["Peclet number"] = self._peclet_number

    def _process_output(self, output, reference_result=False):
        table_string = self._read_convergence_table_string(output)
        convergence_table = self._extract_convergence_table(table_string)
        if reference_result:
            self._reference_table = convergence_table
            self._exactly_equal_columns = set(("cells", "dofs", "dt"))
            self._closely_equal_columns = set(("l2_error", "linfty_error", "h1_error"))

            return True
        else:
            assert hasattr(self, "_reference_table")
            return self._check_convergence_table(self._reference_table,
                                                 convergence_table)


class TGVConvergenceTest(ConvergenceTest):
    def __init__(self, reynolds_number, test_type=TestType.temporal,
                 n_spatial_cycles=2, n_temporal_cycles=2, n_initial_global_refinements=3,
                 n_global_refinements=3, initial_time_step=1e-1, final_time_step=1e-3):
        super().__init__(application_name="TGV",
                         test_type=test_type,
                         n_spatial_cycles=n_spatial_cycles,
                         n_temporal_cycles=n_temporal_cycles,
                         n_initial_global_refinements=n_initial_global_refinements,
                         n_global_refinements=n_global_refinements,
                         initial_time_step=initial_time_step,
                         final_time_step=final_time_step)

        assert isinstance(reynolds_number, float)
        assert reynolds_number > 0.0
        self._reynolds_number = reynolds_number

    def _setup_parameter_mapping(self):
        self._prm_string_value_map["Reynolds number"] = self._reynolds_number

    def _process_output(self, output, reference_result=False):
        table_string = self._read_convergence_table_string(output)
        # split convergence table
        assert self._table_header_str in table_string
        first_table_string = table_string[:table_string.index(self._table_header_str)]
        first_table_string = first_table_string.strip("=")
        second_table_string = table_string[table_string.index(self._table_header_str):]
        second_table_string = second_table_string.strip("=")

        # create pandas object
        first_convergence_table = self._extract_convergence_table(first_table_string)
        second_convergence_table = self._extract_convergence_table(second_table_string)

        if reference_result:
            self._reference_tables = (first_convergence_table,
                                      second_convergence_table)
            self._exactly_equal_columns = set(("cells", "dofs", "dt"))
            self._closely_equal_columns = set(("l2_error", "linfty_error", "h1_error"))

            mandatory_columns = self._exactly_equal_columns.union(self._closely_equal_columns)
            for table in self._reference_tables:
                for col in mandatory_columns:
                    assert col in table

            return True
        else:
            assert hasattr(self, "_reference_tables")

            check_passed = self._check_convergence_table(self._reference_tables[0],
                                                         first_convergence_table)
            check_passed &= self._check_convergence_table(self._reference_tables[1],
                                                          second_convergence_table)
            return check_passed


def compile_library(n_processes=1):
    """
    Compiles the RotatingMHD libary.
    """
    print("Compiling the RotatingMHD libary...", end="", flush=True)
    subprocess.run(["cmake", "CMakeLists.txt"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    subprocess.run(["cmake", "--DCMAKE_BUILD_TYPE=Release", "./"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   text=True, check=True)
    subprocess.run(["make", "-j" + str(n_processes)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    print("   done!")


def postive_integer(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("The argument is not a positive integer.")
    return x


def get_short_commit_hash(commit_hash):
    assert isinstance(commit_hash, str)
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", commit_hash],
                                         text=True)
    return short_hash.strip("\n")


def check_branches(n_processes):
    print("Convergence tests script running with " + str(n_processes) + " processors")

    # define convergence tests
    convergence_tests = []
    thermal_tgv = ThermalTGVConvergenceTest(peclet_number=100.0,
                                            test_type=TestType.temporal,
                                            n_temporal_cycles=3,
                                            n_global_refinements=10,
                                            n_initial_global_refinements=8,
                                            initial_time_step=1.0e-1)
    tgv = TGVConvergenceTest(reynolds_number=100.0,
                             test_type=TestType.temporal,
                             n_temporal_cycles=3,
                             n_global_refinements=10,
                             n_initial_global_refinements=8,
                             initial_time_step=1.0e-1)
    convergence_tests.append(thermal_tgv)
    convergence_tests.append(tgv)

    # define reference branch
    reference_branch = "origin/christensen_benchmark"

    # updates local repository
    subprocess.run(["git", "fetch", "--all"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    # restore current branch
    subprocess.run(["git", "restore", "./"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)

    # checkout to reference branch
    subprocess.run(["git", "checkout", reference_branch],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    current_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True)
    current_hash = current_hash.strip("\n")

    print("{0:25s}: {1}".format("Branch", reference_branch))
    print("{0:25s}: {1}".format("Latest commit", current_hash))

    # run tests on reference branch
    compile_library(n_processes)

    for test in convergence_tests:
        test.run(n_processes, reference_result=True)

    # restore reference branch
    subprocess.run(["git", "restore", "./"], check=True)

    # get all branches merged to master
    branches = subprocess.check_output(["git", "branch", "--remotes",
                                        "--merged", "origin/master"], text=True)
    branches = branches.split(sep="\n")
    branches.remove("  origin/HEAD -> origin/master")
    branches = list(map(lambda x: x.strip(), branches))
    branches.remove("")
#    branches = [reference_branch, "origin/basic_parameters"]

    not_compilable_branches = []
    non_existing_application_branches = dict()
    non_executable_branches = dict()
    incompatible_table_branches = dict()
    infected_branches = dict()
    non_infected_branches = dict()
    for test in convergence_tests:
        non_existing_application_branches[test.name] = []
        non_executable_branches[test.name] = []
        incompatible_table_branches[test.name] = []
        infected_branches[test.name] = []
        non_infected_branches[test.name] = []

    # run tests on branches
    for branch in branches:
        # checkout to current branch
        subprocess.run(["git", "checkout", branch],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        current_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                               text=True)
        current_hash = current_hash.strip("\n")

        # print branch information
        print("{0:25s}: {1}".format("Branch", branch))
        print("{0:25s}: {1}".format("Latest commit", current_hash))

        # run tests on reference branch
        try:
            compile_library(n_processes)
        except subprocess.SubprocessError:
            not_compilable_branches.append(branch)
            continue

        for test in convergence_tests:
            try:
                test_result = test.run(n_processes)
            except ApplicationNotExistingError:
                non_existing_application_branches[test.name].append(branch)
            except ConvergenceTableExtractionError:
                incompatible_table_branches[test.name].append(branch)
            except subprocess.SubprocessError:
                non_executable_branches[test.name].append(branch)

            if test_result is True:
                non_infected_branches[test.name].append(branch)
            else:
                infected_branches[test.name].append(branch)

        # restore current branch
        subprocess.run(["git", "restore", "./"], check=True)

        print("Tests completed!")

    print("Not compilable branches:")
    print(not_compilable_branches)
    print()
    print("Non existing application branches:")
    print(non_existing_application_branches)
    print()
    print("Not executable branches:")
    print(non_executable_branches)
    print()
    print("Infected branches")
    print(infected_branches)
    print()
    print("Non-infected branches")
    print(non_infected_branches)
    print()
    print("Compatible table branches")
    print(incompatible_table_branches)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc", "-np",
                        type=postive_integer,
                        default=4,
                        help="Number of processors")
    args = parser.parse_args()
    check_branches(args.nproc)
