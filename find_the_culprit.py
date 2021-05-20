import argparse
import subprocess
import enum
import fileinput
import numpy
import os

class Test_type(enum.Enum):
  spatial   = 1
  temporal  = 2
  both      = 3



class Dimensionles_number(enum.Enum):
  Reynolds  = 1
  Peclet    = 2



def get_line_number(file, string_to_search):
  line_number = 0
  with open(file, 'r') as read_obj:
    for line in read_obj:
      line_number += 1
      if string_to_search in line:
        break
  return line_number



def replace_line(file, line_number, string_to_replace):
  for line in fileinput.input(file, inplace=True):
    if fileinput.filelineno() == line_number:
      print(string_to_replace)
    else:
      print(line, end="")


class ConvergenceTest:
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
    self.dump_file                    = ""
    self.line_dimensionless_number          = get_line_number(
                                              self.prm_file,
                                              "subsection Dimensionless numbers") + 1
    self.line_convergence_test_type         = get_line_number(
                                              self.prm_file,
                                              "Convergence test type")
    self.line_n_spatial_cycles              = get_line_number(
                                              self.prm_file,
                                              "Number of spatial convergence cycles")
    self.line_n_initial_global_refinements  = get_line_number(
                                              self.prm_file,
                                              "Number of initial global refinements")
    self.line_n_temporal_cycles             = get_line_number(
                                              self.prm_file,
                                              "Number of temporal convergence cycles")
    self.line_initial_time_step             = get_line_number(
                                              self.prm_file,
                                              "Initial time step")
    self.line_graphical_output_directory    = get_line_number(
                                              self.prm_file,
                                              "Graphical output directory")
    self.line_graphical_output_frequency    = get_line_number(
                                              self.prm_file,
                                              "Graphical output frequency")
    replace_line(self.prm_file,
                      self.line_graphical_output_frequency,
                      "  set Graphical output frequency = 10000")

  def extract_convergence_table_from_output(self,
                                            convergence_test_type):
    with open(self.dump_file) as f:
        data = f.readlines()

    if convergence_test_type == Test_type.spatial:
      n_cycles = self.n_spatial_cycles
    elif convergence_test_type == Test_type.temporal:
      n_cycles = self.n_temporal_cycles
    else:
      raise Exception("Invalid enum value.")

    table_start = data.index("==============================================================================================\n") -1

    if self.dimensionless_number_type == Dimensionles_number.Reynolds:
      table_end = table_start + 7 + 2 * n_cycles
    elif self.dimensionless_number_type == Dimensionles_number.Peclet:
      table_end = table_start + 3 + n_cycles

    with open(self.dump_file, 'w') as f:
        f.writelines(data[table_start:table_end])


def nproc_type(x):
  x = int(x)
  if x < 1:
    raise argparse.ArgumentTypeError("Minimum number of processors is 1")
  return x



def run(test, branch_name, hash_number, nproc):
  print("  Source file: " + test.source_file)

  replace_line(test.prm_file,
                test.line_n_spatial_cycles,
              "  set Number of spatial convergence cycles  = %s"
                % test.n_spatial_cycles)
  replace_line(test.prm_file,
                test.line_n_temporal_cycles,
                "  set Number of temporal convergence cycles = %s"
                % test.n_temporal_cycles)

  for dimensionless_number_value in test.dimensionless_number_values:
    if test.dimensionless_number_type == Dimensionles_number.Reynolds:
      replace_line(test.prm_file,
                    test.line_dimensionless_number,
                    "  set Reynolds number         = %s"
                    % dimensionless_number_value)
      print("    Reynolds number: " + str(dimensionless_number_value))
    elif test.dimensionless_number_type == Dimensionles_number.Peclet:
      replace_line(test.prm_file,
                    test.line_dimensionless_number,
                    "  set Peclet number           = %s"
                    % dimensionless_number_value)
      print("    Peclet number: " + str(dimensionless_number_value))
    else:
      raise Exception("Invalid enum value")

    if test.convergence_test_type == Test_type.spatial:
      print("      Running:")
      print("       Convergence test type                 = spatial")
      print("       Number of cycles                      = "
                    + str(test.n_spatial_cycles))
      print("       Number of initial global refinements  = "
                    + str(test.n_initial_global_refinements))
      print("       Time step                             = "
                    + str(test.time_step))

      replace_line(test.prm_file,
            test.line_convergence_test_type,
            "  set Convergence test type                 = spatial")
      replace_line(test.prm_file,
            test.line_n_initial_global_refinements,
            "  set Number of initial global refinements   = %s"
              % test.n_initial_global_refinements)
      replace_line(test.prm_file,
            test.line_initial_time_step,
            "  set Initial time step             = %s"
              % test.time_step)
      replace_line(test.prm_file,
            test.line_graphical_output_directory,
            "   set Graphical output directory = %s_Spatial_%s/"
              % (test.source_file_name, dimensionless_number_value))

      test.dump_file = "Dump/%s_Spatial_%s_%s.txt" \
                          % (test.source_file_name,
                              str(dimensionless_number_value).zfill(4),
                              hash_number)

      with open(test.dump_file, "w") as dump_file:
        process = subprocess.run([test.exec_file, str(nproc)],
                                stdout=dump_file,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

      test.extract_convergence_table_from_output(Test_type.spatial)


    elif test.convergence_test_type == Test_type.temporal:
      print("      Running:")
      print("       Convergence test type                 = temporal")
      print("       Number of cycles                      = "
                    + str(test.n_temporal_cycles))
      print("       Number of global refinements          = "
                    + str(test.n_global_refinements))
      print("       Initial time step                     = "
                    + str(test.initial_time_step))

      replace_line(
            test.prm_file,
            test.line_convergence_test_type,
            "  set Convergence test type                 = temporal")
      replace_line(
            test.prm_file,
            test.line_n_initial_global_refinements,
            "  set Number of initial global refinements   = %s"
              % test.n_global_refinements)
      replace_line(
            test.prm_file,
            test.line_initial_time_step,
            "  set Initial time step             = %s"
              % test.initial_time_step)
      replace_line(
            test.prm_file,
            test.line_graphical_output_directory,
            "  set Graphical output directory = %s_Temporal_%s/"
              % (test.source_file_name, dimensionless_number_value))


      test.dump_file = "Dump/%s_Temporal_%s_%s.txt" \
                          % (test.source_file_name,
                              str(dimensionless_number_value).zfill(4),
                              hash_number)

      with open(test.dump_file, "w") as dump_file:
        process = subprocess.run([test.exec_file, str(nproc)],
                                stdout=dump_file,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

      test.extract_convergence_table_from_output(Test_type.temporal)

    elif test.convergence_test_type == Test_type.both:
      print("      Running:")
      print("       Convergence test type                 = spatial")
      print("       Number of cycles                      = "
                    + str(test.n_spatial_cycles))
      print("       Number of initial global refinements  = "
                    + str(test.n_initial_global_refinements))
      print("       Time step                             = "
                    + str(test.time_step))

      replace_line(
            test.prm_file,
            test.line_convergence_test_type,
            "  set Convergence test type                 = spatial")
      replace_line(
            test.prm_file,
            test.line_n_initial_global_refinements,
            "  set Number of initial global refinements   = %s"
              % test.n_initial_global_refinements)
      replace_line(
            test.prm_file,
            test.line_initial_time_step,
            "  set Initial time step             = %s"
              % test.time_step)
      replace_line(
            test.prm_file,
            test.line_graphical_output_directory,
            "  set Graphical output directory = %s_Spatial_%s/"
              % (test.source_file_name, dimensionless_number_value))

      test.dump_file = "Dump/%s_Spatial_%s_%s.txt" \
                          % (test.source_file_name,
                              str(dimensionless_number_value).zfill(4),
                              hash_number)

      with open(test.dump_file, "w") as dump_file:
        process = subprocess.run([test.exec_file, str(nproc)],
                                stdout=dump_file,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

      test.extract_convergence_table_from_output(Test_type.spatial)

      print("      Running:")
      print("       Convergence test type                 = temporal")
      print("       Number of cycles                      = "
                    + str(test.n_temporal_cycles))
      print("       Number of global refinements          = "
                    + str(test.n_global_refinements))
      print("       Initial time step                     = "
                    + str(test.initial_time_step))

      replace_line(
            test.prm_file,
            test.line_convergence_test_type,
            "  set Convergence test type                 = temporal")
      replace_line(
            test.prm_file,
            test.line_n_initial_global_refinements,
            "  set Number of initial global refinements   = %s"
              % test.n_global_refinements)
      replace_line(
            test.prm_file,
            test.line_initial_time_step,
            "  set Initial time step             = %s"
              % test.initial_time_step)
      replace_line(
            test.prm_file,
            test.line_graphical_output_directory,
            "  set Graphical output directory = %s_Temporal_%s/"
              % (test.source_file_name, dimensionless_number_value))

      test.dump_file = "Dump/%s_Temporal_%s_%s.txt" \
                          % (test.source_file_name,
                              str(dimensionless_number_value).zfill(4),
                              hash_number)

      with open(test.dump_file, "w") as dump_file:
        process = subprocess.run([test.exec_file, str(nproc)],
                                stdout=dump_file,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)

      test.extract_convergence_table_from_output(Test_type.temporal)

    else:
      raise Exception("Invalid enum value")
  print("")


def main(nproc):
  print("Convergence tests script running with " + str(nproc)
        + " processors\n")

  # Updates local repository
  subprocess.run(["git", "fetch", "--all"])

  # Get all the origin/branches
#  branches = subprocess.run(
#                  ["git", "branch", "--remotes", "--merged", "master"],
#                  stdout=subprocess.PIPE,
#                  universal_newlines=True).stdout.split()[2:]
  branches = ["origin/christensen_benchmark", "origin/fix_103"]

  # Get the current branch name
  current_branch = str(subprocess.run(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        stdout=subprocess.PIPE,
                        universal_newlines=True).stdout).strip("\n")

  # Hard reset
  subprocess.run(["git", "reset", "--hard", current_branch])

  # Load tests
  tests = [ConvergenceTest("ThermalTGV",
                           numpy.array([100]),
                           Dimensionles_number.Peclet,
                           Test_type.temporal,
                           2,
                           4,
                           1e-3,
                           2,
                           5,
                           0.1)]


  for branch in branches:
    # Checkout to
    subprocess.run(["git", "checkout", "-f", branch])

    hash = str(subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              stdout=subprocess.PIPE,
                              universal_newlines=True).stdout).strip("\n")

    print(" Branch: " + branch)
    print(" Hash:   " + hash + "\n")

    subprocess.run(["cmake", "CMakeLists.txt"])
    subprocess.run(["cmake", "--DCMAKE_BUILD_TYPE=Release", "./"])
    subprocess.run(["make", "-j"+str(nproc)])

    for test in tests:
      run(test, branch, hash, nproc)

    # Hard reset
    subprocess.run(["git", "reset", "--hard", branch])

  print("Tests completed!")
  print(" The convergence tables are in the applications/*.txt files")
  print(" The terminal output are in the Dump_*.txt files")



if __name__ == "__main__":
  if not os.path.exists("Dump"):
    os.mkdir("Dump")
  parser = argparse.ArgumentParser()
  parser.add_argument("--nproc",
                      "-np",
                      type = nproc_type,
                      default = 4,
                      help = "Number of processors")
  args = parser.parse_args()
  main(args.nproc)
