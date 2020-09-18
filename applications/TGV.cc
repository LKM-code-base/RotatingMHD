/*!
 *@file TGV
 *@brief The .cc file solving the TGV benchmark.
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include <string>
#include <cmath>

namespace RMHD
{
  using namespace dealii;
/*!
 * @class TGV
 * @brief This class solves the Taylor-Green vortex benchmark
 * @todo Add documentation
 */  
template <int dim>
class TGV : public Problem<dim>
{
public:
  TGV(const RunTimeParameters::ParameterSet &parameters);
  void run(const bool         flag_verbose_output           = false,
           const unsigned int terminal_output_periodicity   = 10,
           const unsigned int graphical_output_periodicity  = 10);
private:
  ConditionalOStream                          pcout;

  parallel::distributed::Triangulation<dim>   triangulation;

  std::vector<types::boundary_id>             boundary_ids;

  Entities::VectorEntity<dim>                 velocity;

  Entities::ScalarEntity<dim>                 pressure;

  TimeDiscretization::VSIMEXMethod            time_stepping;

  NavierStokesProjection<dim>                 navier_stokes;

  EquationData::TGV::VelocityExactSolution<dim>         
                                      velocity_exact_solution;

  EquationData::TGV::PressureExactSolution<dim>         
                                      pressure_exact_solution;

  const bool                                  periodic_bcs;

  const bool                                  set_boundary_dofs_to_zero;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_entities();

  void update_boundary_values();
};

template <int dim>
TGV<dim>::TGV(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(),
pcout(std::cout,
      (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
triangulation(MPI_COMM_WORLD,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
velocity(parameters.p_fe_degree + 1, triangulation),
pressure(parameters.p_fe_degree, triangulation),
time_stepping(parameters.time_stepping_parameters),
navier_stokes(parameters, velocity, pressure, time_stepping),
velocity_exact_solution(parameters.Re, parameters.time_stepping_parameters.start_time),
pressure_exact_solution(parameters.Re, parameters.time_stepping_parameters.start_time),
periodic_bcs(true),
set_boundary_dofs_to_zero(false)
{
  make_grid(parameters.n_global_refinements);
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup(true);
  initialize();
}

template <int dim>
void TGV<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  GridGenerator::hyper_cube(triangulation,
                            0.0,
                            2.0*M_PI,
                            true);
  if (periodic_bcs)
  {
    std::vector<GridTools::PeriodicFacePair<
      typename parallel::distributed::Triangulation<dim>::cell_iterator>>
      periodicity_vector;

    FullMatrix<double> rotation_matrix(dim);
    rotation_matrix[0][0] = 1.;
    rotation_matrix[1][1] = 1.;    

    Tensor<1,dim> offset_x;
    offset_x[0] = 2.0 * M_PI;
    offset_x[1] = 0.0;

    Tensor<1,dim> offset_y;
    offset_y[0] = 0.0;
    offset_y[1] = 2.0 * M_PI;

    GridTools::collect_periodic_faces(triangulation,
                                      0,
                                      1,
                                      0,
                                      periodicity_vector,
                                      offset_x,
                                      rotation_matrix);
    GridTools::collect_periodic_faces(triangulation,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector,
                                      offset_y,
                                      rotation_matrix);

    triangulation.add_periodicity(periodicity_vector);
  }

  triangulation.refine_global(n_global_refinements);
  boundary_ids = triangulation.get_boundary_ids();

  pcout     << "Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;
}

template <int dim>
void TGV<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  
  pcout     << "Number of velocity degrees of freedom = " 
            << velocity.dof_handler.n_dofs()
            << std::endl
            << "Number of pressure degrees of freedom = " 
            << pressure.dof_handler.n_dofs()
            << std::endl;
}

template <int dim>
void TGV<dim>::setup_constraints()
{
  {
  velocity.constraints.clear();
  velocity.constraints.reinit(velocity.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(velocity.dof_handler,
                                          velocity.constraints);

  if (periodic_bcs)
  {
    FEValuesExtractors::Vector velocities(0);

    std::vector<unsigned int> first_vector_components;
    first_vector_components.push_back(0);

    std::vector<
    GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

    FullMatrix<double> rotation_matrix(dim);
    rotation_matrix[0][0] = 1.;
    rotation_matrix[1][1] = 1.;

    Tensor<1,dim> offset_x;
    offset_x[0] = 2.0 * M_PI;
    offset_x[1] = 0.0;

    Tensor<1,dim> offset_y;
    offset_y[0] = 0.0;
    offset_y[1] = 2.0 * M_PI;

    GridTools::collect_periodic_faces(velocity.dof_handler,
                                      0,
                                      1,
                                      0,
                                      periodicity_vector,
                                      offset_x,
                                      rotation_matrix);
    GridTools::collect_periodic_faces(velocity.dof_handler,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector,
                                      offset_y,
                                      rotation_matrix);

    DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
      periodicity_vector,
      velocity.constraints,
      velocity.fe.component_mask(velocities),
      first_vector_components);
  }
  else
  {
    VectorTools::interpolate_boundary_values(
                                  velocity.dof_handler,
                                  0,
                                  velocity_exact_solution,
                                  velocity.constraints);
    VectorTools::interpolate_boundary_values(
                                  velocity.dof_handler,
                                  1,
                                  velocity_exact_solution,
                                  velocity.constraints);
    VectorTools::interpolate_boundary_values(
                                  velocity.dof_handler,
                                  2,
                                  velocity_exact_solution,
                                  velocity.constraints);
    VectorTools::interpolate_boundary_values(
                                  velocity.dof_handler,
                                  3,
                                  velocity_exact_solution,
                                  velocity.constraints);
  }

  velocity.constraints.close();
  }

  {
  pressure.constraints.clear();
  pressure.constraints.reinit(pressure.locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(pressure.dof_handler,
                                          pressure.constraints);
  if (periodic_bcs)
  {
    FEValuesExtractors::Scalar pressure_extractor(0);

    std::vector<unsigned int> first_vector_components;
    first_vector_components.push_back(0);

    std::vector<
    GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
      periodicity_vector;

    FullMatrix<double> rotation_matrix(dim);
    rotation_matrix[0][0] = 1.;
    rotation_matrix[1][1] = 1.;

    Tensor<1,dim> offset_x;
    offset_x[0] = 2.0 * M_PI;
    offset_x[1] = 0.0;

    Tensor<1,dim> offset_y;
    offset_y[0] = 0.0;
    offset_y[1] = 2.0 * M_PI;

    GridTools::collect_periodic_faces(pressure.dof_handler,
                                      0,
                                      1,
                                      0,
                                      periodicity_vector,
                                      offset_x,
                                      rotation_matrix);
    GridTools::collect_periodic_faces(pressure.dof_handler,
                                      2,
                                      3,
                                      1,
                                      periodicity_vector,
                                      offset_y,
                                      rotation_matrix);

    DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
      periodicity_vector,
      pressure.constraints,
      pressure.fe.component_mask(pressure_extractor),
      first_vector_components);
  }
  if (set_boundary_dofs_to_zero)
  {
    const FEValuesExtractors::Scalar    pressure_extractor(0);

    IndexSet    boundary_pressure_dofs;

    DoFTools::extract_boundary_dofs(pressure.dof_handler,
                                    pressure.fe.component_mask(pressure_extractor),
                                    boundary_pressure_dofs);

    types::global_dof_index local_idx = numbers::invalid_dof_index;
    
    IndexSet::ElementIterator
    idx = boundary_pressure_dofs.begin(),
    endidx = boundary_pressure_dofs.end();
    for(; idx != endidx; ++idx)
        if (pressure.constraints.can_store_line(*idx)
                && !pressure.constraints.is_constrained(*idx))
            local_idx = *idx;

    const types::global_dof_index global_idx
    = Utilities::MPI::min(
            (local_idx != numbers::invalid_dof_index) ?
                    local_idx :
                    pressure.dof_handler.n_dofs(),
            pressure.mpi_communicator);

    Assert(global_idx < pressure.dof_handler.n_dofs(),
            ExcMessage("Error, couldn't find a pressure DoF to constrain."));

    // Finally set this DoF to zero (if we care about it):
    if (pressure.constraints.can_store_line(global_idx))
    {
        Assert(!pressure.constraints.is_constrained(global_idx), ExcInternalError());
        pressure.constraints.add_line(global_idx);
    }
  }
  pressure.constraints.close();
  }
}

template <int dim>
void TGV<dim>::initialize()
{
  this->set_initial_conditions(velocity, 
                               velocity_exact_solution, 
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_exact_solution, 
                               time_stepping);
  
  //navier_stokes.initialize();
}

template <int dim>
void TGV<dim>::postprocessing(const bool flag_point_evaluation)
{
  if (flag_point_evaluation)
  {
    pcout << "Step = " 
          << std::setw(4) 
          << time_stepping.get_step_number() 
          << " Time = " 
          << std::noshowpos << std::scientific
          << time_stepping.get_next_time()    
          << std::endl;
  }
}

template <int dim>
void TGV<dim>::output()
{
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;
  data_out.add_data_vector(velocity.dof_handler,
                           velocity.solution,
                           names, 
                           component_interpretation);
  data_out.add_data_vector(pressure.dof_handler, 
                           pressure.solution, 
                           "Pressure");
  data_out.build_patches(velocity.fe_degree);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

template <int dim>
void TGV<dim>::update_entities()
{
  velocity.update_solution_vectors();
  pressure.update_solution_vectors();
  navier_stokes.update_internal_entities();
  velocity_exact_solution.set_time(time_stepping.get_next_time());
  pressure_exact_solution.set_time(time_stepping.get_next_time());
  update_boundary_values();
}

template <int dim>
void TGV<dim>::update_boundary_values()
{
  if (!periodic_bcs)
  {
    AffineConstraints<double>     tmp_constraints;
    tmp_constraints.clear();
    tmp_constraints.reinit(velocity.locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(velocity.dof_handler,
                                            tmp_constraints);
    VectorTools::interpolate_boundary_values(
                                velocity.dof_handler,
                                0,
                                velocity_exact_solution,
                                tmp_constraints);
    VectorTools::interpolate_boundary_values(
                                velocity.dof_handler,
                                1,
                                velocity_exact_solution,
                                tmp_constraints);
    VectorTools::interpolate_boundary_values(
                                velocity.dof_handler,
                                2,
                                velocity_exact_solution,
                                tmp_constraints);
    VectorTools::interpolate_boundary_values(
                                velocity.dof_handler,
                                3,
                                velocity_exact_solution,
                                tmp_constraints);
    tmp_constraints.close();
    velocity.constraints.merge(
      tmp_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  }
}

template <int dim>
void TGV<dim>::run(
              const bool          flag_verbose_output,
              const unsigned int  terminal_output_periodicity,
              const unsigned int  graphical_output_periodicity)
{
  (void)flag_verbose_output;

  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();
  
  velocity.solution = velocity.old_old_solution;
  pressure.solution = pressure.old_old_solution;
  output();
  velocity.solution = velocity.old_solution;
  pressure.solution = pressure.old_solution;
  output();

  velocity_exact_solution.set_time(time_stepping.get_next_time());
  pressure_exact_solution.set_time(time_stepping.get_next_time());
  update_boundary_values();

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    time_stepping.set_desired_next_step_size(
                              navier_stokes.compute_next_time_step());

    time_stepping.update_coefficients();

    navier_stokes.solve(time_stepping.get_step_number());

    // snapshot stage
    postprocessing((time_stepping.get_step_number() %
                    terminal_output_periodicity == 0) ||
                    (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          graphical_output_periodicity == 0) ||
        (time_stepping.get_next_time() == 
          time_stepping.get_end_time()))
      output();
    
    update_entities();
    time_stepping.advance_time();
  }
}

} // namespace RMHD

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace RMHD;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

      RunTimeParameters::ParameterSet parameter_set("TGV.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      TGV<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_verbose_output, 
                     parameter_set.terminal_output_interval,
                     parameter_set.graphical_output_interval);
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
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}
