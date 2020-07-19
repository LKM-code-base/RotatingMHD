/*
 * data_storage.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */


#include <rotatingMHD/run_time_parameters.h>

#include <fstream>

namespace Step35
{

using namespace dealii;

namespace RunTimeParameters
{

// In the constructor of this class we declare all the parameters. The
// details of how this works have been discussed elsewhere, for example in
// step-19 and step-29.
Data_Storage::Data_Storage()
:
form(Method::rotational),
dt(5e-4),
initial_time(0.),
final_time(1.),
Reynolds(1.),
n_global_refines(0),
pressure_degree(1),
vel_max_iterations(1000),
vel_Krylov_size(30),
vel_off_diagonals(60),
vel_update_prec(15),
vel_eps(1e-12),
vel_diag_strength(0.01),
verbose(true),
output_interval(15)
{
	prm.declare_entry("Method_Form",
	                  "rotational",
	                  Patterns::Selection("rotational|standard"),
	                  " Used to select the type of method that we are going "
	                  "to use. ");
	prm.enter_subsection("Physical data");
	{
	  prm.declare_entry("initial_time",
	                    "0.",
	                    Patterns::Double(0.),
	                    " The initial time of the simulation. ");
	  prm.declare_entry("final_time",
	                    "1.",
	                    Patterns::Double(0.),
	                    " The final time of the simulation. ");
		prm.declare_entry("Reynolds",
				              "1.",
				              Patterns::Double(0.),
				              " The Reynolds number. ");
	}
	prm.leave_subsection();

	prm.enter_subsection("Time step data");
	{
	  prm.declare_entry("dt",
				              "5e-4",
				              Patterns::Double(0.),
				              " The time step size. ");
	}
	prm.leave_subsection();

	prm.enter_subsection("Space discretization");
	{
		prm.declare_entry("n_of_refines",
				              "0",
				              Patterns::Integer(0, 15),
				              " The number of global refines we do on the mesh. ");
		prm.declare_entry("pressure_fe_degree",
				              "1",
				              Patterns::Integer(1, 5),
				              " The polynomial degree for the pressure space. ");
	}
	prm.leave_subsection();

	prm.enter_subsection("Data solve velocity");
	{
		prm.declare_entry("max_iterations",
				              "1000",
				              Patterns::Integer(1, 1000),
				              " The maximal number of iterations GMRES must make. ");
		prm.declare_entry("eps",
				              "1e-12",
				              Patterns::Double(0.),
				              " The stopping criterion. ");
		prm.declare_entry("Krylov_size",
				              "30",
				              Patterns::Integer(1),
				              " The size of the Krylov subspace to be used. ");
		prm.declare_entry("off_diagonals",
				              "60",
				              Patterns::Integer(0),
				              " The number of off-diagonal elements ILU must "
				              "compute. ");
		prm.declare_entry("diag_strength",
				              "0.01",
				              Patterns::Double(0.),
				              " Diagonal strengthening coefficient. ");
		prm.declare_entry("update_prec",
				              "15",
				              Patterns::Integer(1),
				              " This number indicates how often we need to "
				              "update the preconditioner");
	}
	prm.leave_subsection();

	prm.declare_entry("verbose",
			              "true",
			              Patterns::Bool(),
			              " This indicates whether the output of the solution "
			              "process should be verbose. ");

	prm.declare_entry("output_interval",
			              "1",
			              Patterns::Integer(1),
			              " This indicates between how many time steps we print "
			              "the solution. ");
}

void Data_Storage::read_data(const std::string &filename)
{
  std::ifstream file(filename);
  AssertThrow(file, ExcFileNotOpen(filename));

  prm.parse_input(file);

  if (prm.get("Method_Form") == std::string("rotational"))
    form = Method::rotational;
  else
    form = Method::standard;

  prm.enter_subsection("Physical data");
  {
    initial_time = prm.get_double("initial_time");
    final_time   = prm.get_double("final_time");
    Reynolds     = prm.get_double("Reynolds");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time step data");
  {
    dt = prm.get_double("dt");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space discretization");
  {
    n_global_refines = prm.get_integer("n_of_refines");
    pressure_degree  = prm.get_integer("pressure_fe_degree");
  }
  prm.leave_subsection();

  prm.enter_subsection("Data solve velocity");
  {
    vel_max_iterations = prm.get_integer("max_iterations");
    vel_eps            = prm.get_double("eps");
    vel_Krylov_size    = prm.get_integer("Krylov_size");
    vel_off_diagonals  = prm.get_integer("off_diagonals");
    vel_diag_strength  = prm.get_double("diag_strength");
    vel_update_prec    = prm.get_integer("update_prec");
  }
  prm.leave_subsection();

  verbose = prm.get_bool("verbose");

  output_interval = prm.get_integer("output_interval");
}

} // namespace RunTimeParameters

} // namespace Step35
