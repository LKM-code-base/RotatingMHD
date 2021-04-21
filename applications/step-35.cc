
#include <rotatingMHD/hydrodynamic_problem.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/equation_data.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>

namespace RMHD
{

using namespace dealii;

template <int dim>
class Step35 : public HydrodynamicProblem<dim>
{
public:
  Step35(RunTimeParameters::HydrodynamicProblemParameters &parameters);

  void run();

private:
  const Point<dim>  evaluation_point;

  const types::boundary_id  wall_boundary_id;
  const types::boundary_id  inlet_boundary_id;
  const types::boundary_id  outlet_boundary_id;
  const types::boundary_id  square_boundary_id;

  void sample_point_data(const Point<dim> &point) const;

  virtual void make_grid() override;

  virtual void postprocess_solution() override;

  virtual void setup_boundary_conditions() override;

  virtual void setup_initial_conditions() override;
};

template <int dim>
Step35<dim>::Step35
(RunTimeParameters::HydrodynamicProblemParameters &parameters)
:
HydrodynamicProblem<dim>(parameters),
evaluation_point(2.0, 3.0),
wall_boundary_id(1),
inlet_boundary_id(2),
outlet_boundary_id(3),
square_boundary_id(4)
{}

template<int dim>
void Step35<dim>::run()
{
  HydrodynamicProblem<dim>::run();
}

template <int dim>
void Step35<dim>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);

  {
    std::string   filename = "nsbench2.inp";

    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));

    grid_in.read_ucd(file);

    file.close();
  }

  this->triangulation.refine_global(this->prm.spatial_discretization_parameters.n_initial_global_refinements);

  *(this->pcout) << "Number of active cells                = "
                 << this->triangulation.n_global_active_cells() << std::endl;
}

template <int dim>
void Step35<dim>::setup_boundary_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  this->velocity->boundary_conditions.clear();
  this->pressure->boundary_conditions.clear();

  const double current_time = this->time_stepping.get_current_time();
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Boundary conditions are not setup at the start time."));

  this->velocity->boundary_conditions.set_dirichlet_bcs(wall_boundary_id);

  using namespace EquationData::Step35;
  this->velocity->boundary_conditions.set_dirichlet_bcs
  (inlet_boundary_id,
   std::make_shared<VelocityInflowBoundaryCondition<dim>>(current_time));

  this->velocity->boundary_conditions.set_dirichlet_bcs(square_boundary_id);
  this->velocity->boundary_conditions.set_tangential_flux_bcs(outlet_boundary_id);

  this->pressure->boundary_conditions.set_dirichlet_bcs(outlet_boundary_id);

  this->velocity->apply_boundary_conditions();

  this->pressure->apply_boundary_conditions();

}

template <int dim>
void Step35<dim>::setup_initial_conditions()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  const double current_time = this->time_stepping.get_current_time();
  Assert(current_time == this->time_stepping.get_start_time(),
         ExcMessage("Initial conditions are not setup at the start time."));

  using namespace EquationData::Step35;
  const VelocityInitialCondition<dim>  velocity_initial_condition(dim);
  this->project_function(velocity_initial_condition,
                         this->velocity,
                         this->velocity->old_solution);
  this->velocity->solution = this->velocity->old_solution;

  const PressureInitialCondition<dim> pressure_initial_condition(current_time);
  this->project_function(pressure_initial_condition,
                         this->pressure,
                         this->pressure->old_solution);
  this->pressure->solution = this->pressure->old_solution;
}

template <int dim>
void Step35<dim>::postprocess_solution()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  const Tensor<1,dim> velocity_value = this->velocity->point_value(evaluation_point);

  *this->pcout << "   " << "Velocity = ("
               << velocity_value
               << "); ";

  const double pressure_value = this->pressure->point_value(evaluation_point);
  *this->pcout << "Pressure = "
               << Utilities::to_string(pressure_value, 4)
               << std::endl;
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace RMHD;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    RunTimeParameters::HydrodynamicProblemParameters parameter_set("step-35.prm");

    Step35<2> simulation(parameter_set);
    simulation.run();

  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  return 0;
}
