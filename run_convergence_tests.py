import argparse     # Module to parse arguments
import subprocess   # Module to execute terminal commands
import enum         # Module to enable a enum class
import fileinput    # Module to modify .txt files
import numpy



class Test_type(enum.Enum):
  """Test_type Enum class for the type of convergence test to be done
  """
  spatial   = 1
  temporal  = 2
  both      = 3



class Dimensionles_number(enum.Enum):
  """Dimensionles_number Enum class for the relevant dimensionless
  numbers
  """
  Reynolds  = 1
  Peclet    = 2



class ConvergenceTest:
  """Class encompasing all the data related to the convergence test.

  The class stores the convergence tests parameters passed on to the
  constructor and the pertinent line numbers using the `get_line_number`
  function

  :param source_file_name: The name of the source file
  :type source_file_name: str
  :param dimensionless_number_values: A numpy array containing the values
    of the dimensionless number for which a convergence test is
    to be performed, defaults to numpy.array([100])
  :type dimensionless_number_values: numpy.array, optional
  :param dimensionless_number_type: The pertinent dimensionless number
    of the problem (Reynolds or Peclet), defaults to
    Dimensionles_number.Reynolds
  :type dimensionless_number_type: class.Dimensionless_number, optional
  :param convergence_test_type: The type of convergence test to be done,
    defaults to Test_type.temporal
  :type convergence_test_type: class.Test_type, optional
  :param n_spatial_cycles: The number of cycles to be performed in the
    spatial convergence test, defaults to 2
  :type n_spatial_cycles: int, optional
  :param n_initial_global_refinements: The initial number of global
    refinements, defaults to 5
  :type n_initial_global_refinements: int, optional
  :param time_step: The fixed time step size used throughout the spatial
    convergence test, defaults to 1e-3
  :type time_step: float, optional
  :param n_temporal_cycles: The number of cycles to be performed in the
    temporal convergence test, defaults to 2
  :type n_temporal_cycles: int, optional
  :param n_global_refinements: The fixed number of global refinements
    used throughout the temporal convergence test, defaults to 8
  :type n_global_refinements: int, optional
  :param initial_time_step: The initial time step size, defaults to 1e-1
  :type initial_time_step: float, optional
  :param exec_file: The shell script which actually runs the convergence
    test.
  :type exec_file: str
  :param source_file: The source file
  :type source_file: str
  :param prm_file: The .prm file
  :type prm_file: str
  :param line_dimensionless_number: The line number of the .prm file
    at which the pertinent dimensionless number is set
  :type line_dimensionless_number: int
  :param line_convergence_test_type: The line number of the .prm file
    at which the convergence test type is set
  :type line_convergence_test_type: int
  :param line_n_spatial_cycles: The line number of the .prm file
    at which number of spatial cycles is set
  :type line_n_spatial_cycles: int
  :param line_n_initial_global_refinements: The line number of the .prm
    file at which the initial number of global refinements is set
  :type line_n_initial_global_refinements: int
  :param line_n_temporal_cycles: The line number of the .prm file
    at which the number of temporal cycles is set
  :type line_n_temporal_cycles: int
  :param line_initial_time_step: The line number of the .prm file
    at which the initial time step size is set
  :type line_initial_time_step: int
  :param line_graphical_output_directory: The line number of the .prm
    file at which graphical output directory is set
  :type line_graphical_output_directory: int
  :param line_graphical_output_frequency: The line number of the .prm
    file at which graphical output frequency is set
  :type line_graphical_output_frequency: int

  """
  def __init__(self,
               source_file_name,
               dimensionless_number_values  = numpy.array([100]),
               dimensionless_number_type    = Dimensionles_number.Reynolds,
               convergence_test_type        = Test_type.temporal,
               n_spatial_cycles             = 2,
               n_initial_global_refinements = 5,
               time_step                    = 1e-3,
               n_temporal_cycles            = 2,
               n_global_refinements         = 8,
               initial_time_step            = 1e-1):
    """Contructor method

    Stores the arguments passed on to the constructor in the class'
    attributes. Determines the line numbers of the relevant data inside
    the .prm file using the `get_line_number` function and stores them
    in the class' attributes.

    """
    self.source_file_name             = source_file_name
    self.dimensionless_number_values  = dimensionless_number_values
    self.dimensionless_number_type    = dimensionless_number_type
    self.convergence_test_type        = convergence_test_type
    self.n_spatial_cycles             = n_spatial_cycles
    self.n_initial_global_refinements = n_initial_global_refinements
    self.time_step                    = time_step
    self.n_temporal_cycles            = n_temporal_cycles
    self.initial_time_step            = initial_time_step
    self.n_global_refinements         = n_global_refinements
    self.exec_file                    = "./" + source_file_name + ".sh"
    self.source_file                  = source_file_name + ".cc"
    self.prm_file                     = "applications/" + source_file_name + ".prm"
    self.line_dimensionless_number          = self.get_line_number(
                                              "subsection Dimensionless numbers") + 1
    self.line_convergence_test_type         = self.get_line_number(
                                              "Convergence test type")
    self.line_n_spatial_cycles              = self.get_line_number(
                                              "Number of spatial convergence cycles")
    self.line_n_initial_global_refinements  = self.get_line_number(
                                              "Number of initial global refinements")
    self.line_n_temporal_cycles             = self.get_line_number(
                                              "Number of temporal convergence cycles")
    self.line_initial_time_step             = self.get_line_number(
                                              "Initial time step")
    self.line_graphical_output_directory    = self.get_line_number(
                                              "Graphical output directory")
    self.line_graphical_output_frequency    = self.get_line_number(
                                              "Graphical output frequency")
    self.replace_line(self.line_graphical_output_frequency,
                      "  set Graphical output frequency = 10000")

  def get_line_number(self, string_to_search):
    """get_line_number Locates the `string_to_search` in the `prm_file`
    and returns its line number

    :param string_to_search: String to search in the `prm_file`
    :type string_to_search: str
    :return: Line number where `string_to_search` is found
    :rtype: int

    """
    line_number = 0
    with open(self.prm_file, 'r') as read_obj:
      for line in read_obj:
        line_number += 1
        if string_to_search in line:
          break
    return line_number

  def replace_line(self, line_number, string_to_replace):
    """replace_line Replaces the `line_number`-th line of the `prm_file`
    with `string_to_replace`

    :param line_number: The number of the line to be overwritten
    :type line_number: int
    :param string_to_replace: String which overwrites the
      `line_number`-th line
    :type string_to_replace: str

    """
    for line in fileinput.input(self.prm_file, inplace=True):
      if fileinput.filelineno() == line_number:
        print(string_to_replace)
      else:
        print(line, end="")



def nproc_type(x):
  """nproc_type A function acting as a type for admissible number of
  processors, i.e. only positive integers bigger than zero

  :param x: Desired number of processors
  :type x: int
  :raises argparse.ArgumentTypeError: Error if `x` is smaller than 1
  :return: Admissible number of proccesors
  :rtype: int

  """
  x = int(x)
  if x < 1:
    raise argparse.ArgumentTypeError("Minimum number of processors is 1")
  return x



def main(nproc):
  branch  = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                           stdout=subprocess.PIPE,
                           universal_newlines=True)
  hash    = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           stdout=subprocess.PIPE,
                           universal_newlines=True)
  print("Convergence tests script running with " + str(nproc)
        + " processors")
  print(" Branch: " + str(branch.stdout).strip("\n"))
  print(" Hash:   " + str(hash.stdout))

  # Load all the convergence tests and its setting into a list
  tests = [ConvergenceTest("ThermalTGV",
                           numpy.array([100]),
                           Dimensionles_number.Peclet,
                           Test_type.temporal,
                           2,
                           4,
                           1e-3,
                           2,
                           5,
                           0.1),
           ConvergenceTest("TGV",
                           numpy.array([100]),
                           Dimensionles_number.Reynolds,
                           Test_type.temporal,
                           2,
                           4,
                           1e-3,
                           2,
                           5,
                           0.1)]

  # Loop over the list
  for test in tests:
    print("  Source file: " + test.source_file)

    # Set the tests' number of cycles in the .prm file
    test.replace_line(test.line_n_spatial_cycles,
                      "  set Number of spatial convergence cycles  = %s"
                        % test.n_spatial_cycles)
    test.replace_line(test.line_n_temporal_cycles,
                      "  set Number of temporal convergence cycles = %s"
                        % test.n_temporal_cycles)

    # Loops over the values of the dimensionless number
    for dimensionless_number_value in test.dimensionless_number_values:
      # Set the test's dimensionless value in the .prm file
      if test.dimensionless_number_type == Dimensionles_number.Reynolds:
        test.replace_line(test.line_dimensionless_number,
                          "  set Reynolds number         = %s"
                            % dimensionless_number_value)
        print("    Reynolds number: " + str(dimensionless_number_value))
      elif test.dimensionless_number_type == Dimensionles_number.Peclet:
        test.replace_line(test.line_dimensionless_number,
                          "  set Peclet number           = %s"
                            % dimensionless_number_value)
        print("    Peclet number: " + str(dimensionless_number_value))
      else:
        raise Exception("Invalid enum value")

      # if-branches depending on the type of convergence test. Only the
      # first one is commented as the all follow the same pattern
      if test.convergence_test_type == Test_type.spatial:
        # Terminal output
        print("      Running:")
        print("       Convergence test type                 = spatial")
        print("       Number of cycles                      = "
                      + str(test.n_spatial_cycles))
        print("       Number of initial global refinements  = "
                      + str(test.n_initial_global_refinements))
        print("       Time step                             = "
                      + str(test.time_step))

        # Set the pertinent values in the .prm file
        test.replace_line(
              test.line_convergence_test_type,
              "  set Convergence test type                 = spatial")
        test.replace_line(
              test.line_n_initial_global_refinements,
              "  set Number of initial global refinements   = %s"
                % test.n_initial_global_refinements)
        test.replace_line(
              test.line_initial_time_step,
              "  set Initial time step             = %s"
                % test.time_step)
        test.replace_line(
              test.line_graphical_output_directory,
              "   set Graphical output directory = %s_Spatial_%s/"
                % (test.source_file_name, dimensionless_number_value))

        # Run the shell script
        with open("Dump_%s_Spatial_%s.txt"
                    % (test.source_file_name,
                       dimensionless_number_value),
                  "w") as dump_file:
          process = subprocess.run([test.exec_file, str(nproc)],
                                  stdout=dump_file,
                                  stderr=subprocess.STDOUT)


      elif test.convergence_test_type == Test_type.temporal:
        print("      Running:")
        print("       Convergence test type                 = temporal")
        print("       Number of cycles                      = "
                      + str(test.n_temporal_cycles))
        print("       Number of global refinements          = "
                      + str(test.n_global_refinements))
        print("       Initial time step                     = "
                      + str(test.initial_time_step))

        test.replace_line(
              test.line_convergence_test_type,
              "  set Convergence test type                 = temporal")
        test.replace_line(
              test.line_n_initial_global_refinements,
              "  set Number of initial global refinements   = %s"
                % test.n_global_refinements)
        test.replace_line(
              test.line_initial_time_step,
              "  set Initial time step             = %s"
                % test.initial_time_step)
        test.replace_line(
              test.line_graphical_output_directory,
              "  set Graphical output directory = %s_Temporal_%s/"
                % (test.source_file_name, dimensionless_number_value))

        with open("Dump_%s_Temporal_%s.txt"
                    % (test.source_file_name,
                       dimensionless_number_value),
                  "w") as dump_file:
          process = subprocess.run([test.exec_file, str(nproc)],
                                  stdout=dump_file,
                                  stderr=subprocess.STDOUT)

      elif test.convergence_test_type == Test_type.both:
        print("      Running:")
        print("       Convergence test type                 = spatial")
        print("       Number of cycles                      = "
                      + str(test.n_spatial_cycles))
        print("       Number of initial global refinements  = "
                      + str(test.n_initial_global_refinements))
        print("       Time step                             = "
                      + str(test.time_step))

        test.replace_line(
              test.line_convergence_test_type,
              "  set Convergence test type                 = spatial")
        test.replace_line(
              test.line_n_initial_global_refinements,
              "  set Number of initial global refinements   = %s"
                % test.n_initial_global_refinements)
        test.replace_line(
              test.line_initial_time_step,
              "  set Initial time step             = %s"
                % test.time_step)
        test.replace_line(
              test.line_graphical_output_directory,
              "  set Graphical output directory = %s_Spatial_%s/"
                % (test.source_file_name, dimensionless_number_value))

        with open("Dump_%s_Spatial_%s.txt"
                    % (test.source_file_name,
                       dimensionless_number_value),
                  "w") as dump_file:
          process = subprocess.run([test.exec_file, str(nproc)],
                                  stdout=dump_file,
                                  stderr=subprocess.STDOUT)

        print("      Running:")
        print("       Convergence test type                 = temporal")
        print("       Number of cycles                      = "
                      + str(test.n_temporal_cycles))
        print("       Number of global refinements          = "
                      + str(test.n_global_refinements))
        print("       Initial time step                     = "
                      + str(test.initial_time_step))

        test.replace_line(
                test.line_convergence_test_type,
                "  set Convergence test type                 = temporal")
        test.replace_line(
              test.line_n_initial_global_refinements,
              "  set Number of initial global refinements   = %s"
                % test.n_global_refinements)
        test.replace_line(
              test.line_initial_time_step,
              "  set Initial time step             = %s"
                % test.initial_time_step)
        test.replace_line(
                test.line_graphical_output_directory,
                "  set Graphical output directory = %s_Temporal_%s/"
                  % (test.source_file_name, dimensionless_number_value))

        with open("Dump_%s_Temporal_%s.txt"
                    % (test.source_file_name,
                       dimensionless_number_value),
                  "w") as dump_file:
          process = subprocess.run([test.exec_file, str(nproc)],
                                  stdout=dump_file,
                                  stderr=subprocess.STDOUT)

      else:
        raise Exception("Invalid enum value")
    print("")
  print("Tests completed!")
  print(" The convergence tables are in the applications/*.txt files")
  print(" The terminal output are in the Dump_*.txt files")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--nproc",
                      "-np",
                      type = nproc_type,
                      default = 4,
                      help = "Number of processors")
  args = parser.parse_args()
  main(args.nproc)