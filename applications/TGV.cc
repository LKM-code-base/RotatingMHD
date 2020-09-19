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
#include <deal.II/base/convergence_table.h>
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

template <int dim>
struct ConvergenceAnalysisData
{
  ConvergenceTable                convergence_table;

  const std::string               entity_name;

  const Entities::EntityBase<dim> &entity;

  const Function<dim>             &exact_solution;

  ConvergenceAnalysisData(const Entities::EntityBase<dim> &entity,
                          const Function<dim>             &exact_solution,
                          const std::string entity_name = "Entity")
  :
  entity_name(entity_name),
  entity(entity),
  exact_solution(exact_solution)
  {
    convergence_table.declare_column("level");
    convergence_table.declare_column("dt");
    convergence_table.declare_column("cells");
    convergence_table.declare_column("dofs");
    convergence_table.declare_column("hmax");
    convergence_table.declare_column("L2");
    convergence_table.declare_column("Linfty");
    //convergence_table.declare_column("H1");
    convergence_table.set_scientific("dt", true);
    convergence_table.set_scientific("hmax", true);
    convergence_table.set_scientific("L2", true);
    //convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
    convergence_table.set_precision("dt", 2);
    convergence_table.set_precision("hmax", 2);
    convergence_table.set_precision("L2", 6);
    //convergence_table.set_precision("H1", 6);
    convergence_table.set_precision("Linfty", 6);
  }

  void update_table(const unsigned int  &level,
                    const double        &time_step,
                    const bool          &flag)
  {
  Vector<double> cellwise_difference(
    entity.dof_handler.get_triangulation().n_active_cells());

  QGauss<dim>    quadrature_formula(entity.fe_degree + 2);
  const QTrapez<1>     trapezoidal_rule;
  const QIterated<dim> iterated_quadrature_rule(trapezoidal_rule,
                                                entity.fe_degree * 2 + 1);
  
  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::L2_norm);
  
  const double L2_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::L2_norm);
/*
  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::H1_norm);
  
  const double H1_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::H1_norm);*/

  VectorTools::integrate_difference(entity.dof_handler,
                                    entity.solution,
                                    exact_solution,
                                    cellwise_difference,
                                    iterated_quadrature_rule,
                                    VectorTools::Linfty_norm);
  
  const double Linfty_error =
    VectorTools::compute_global_error(entity.dof_handler.get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::Linfty_norm);

  convergence_table.add_value("level", level);
  convergence_table.add_value("dt", time_step);
  convergence_table.add_value("cells", entity.dof_handler.get_triangulation().n_active_cells());
  convergence_table.add_value("dofs", entity.dof_handler.n_dofs());
  convergence_table.add_value("hmax", GridTools::maximal_cell_diameter(entity.dof_handler.get_triangulation()));
  convergence_table.add_value("L2", L2_error);
  //convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

  std::string reference_column = (flag) ? "hmax" : "dt";

  convergence_table.evaluate_convergence_rates(
                              "L2",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  /*convergence_table.evaluate_convergence_rates(
                              "H1",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);*/
  convergence_table.evaluate_convergence_rates(
                              "Linfty",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  }

  void print_table()
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
    std::cout << std::endl;
    std::cout 
      << "                    " << entity_name 
      << " convergence table" << std::endl
      << "===============================================================" 
      << std::endl;
    convergence_table.write_text(std::cout);
    }
  }
};

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
  void run(const bool flag_convergence_test);
private:
  const RunTimeParameters::ParameterSet             &prm;

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

  ConvergenceAnalysisData<dim>                velocity_convergence_table;

  ConvergenceAnalysisData<dim>                pressure_convergence_table;

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

  void solve(const unsigned int &level);
};

template <int dim>
TGV<dim>::TGV(const RunTimeParameters::ParameterSet &parameters)
:
Problem<dim>(),
prm(parameters),
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
velocity_convergence_table(velocity, velocity_exact_solution, "Velocity"),
pressure_convergence_table(pressure, pressure_exact_solution, "Pressure"),
periodic_bcs(true),
set_boundary_dofs_to_zero(false)
{}

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
}

template <int dim>
void TGV<dim>::setup_dofs()
{
  velocity.setup_dofs();
  pressure.setup_dofs();
  pcout     << "  Number of active cells                = " 
            << triangulation.n_active_cells() << std::endl;
  pcout     << "  Number of velocity degrees of freedom = " 
            << velocity.dof_handler.n_dofs()
            << std::endl
            << "  Number of pressure degrees of freedom = " 
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
    for (const auto& boundary_id : boundary_ids)
      VectorTools::interpolate_boundary_values(
                                    velocity.dof_handler,
                                    boundary_id,
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
  // The diffusion prestep does not produce physical results.
  // Could this be due to the boundary conditions? Since there is no
  // body force in this problem the problem reduces to a boundary integral
  // whhose integrand is projected in the normal to the boundary direction.
  //navier_stokes.initialize();
}

template <int dim>
void TGV<dim>::postprocessing(const bool flag_point_evaluation)
{
  if (flag_point_evaluation)
  {
    std::cout.precision(1);
    pcout << "  Step = " 
          << std::setw(4) 
          << time_stepping.get_step_number() 
          << " Time = " 
          << std::noshowpos << std::scientific
          << time_stepping.get_next_time()
          << " Progress ["
          << std::setw(5) 
          << std::fixed
          << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
          << "%] \r";
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
    for (const auto& boundary_id : boundary_ids)
      VectorTools::interpolate_boundary_values(
                                  velocity.dof_handler,
                                  boundary_id,
                                  velocity_exact_solution,
                                  tmp_constraints);
    tmp_constraints.close();
    velocity.constraints.merge(
      tmp_constraints,
      AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  }
}

template <int dim>
void TGV<dim>::solve(const unsigned int &level)
{
  setup_dofs();
  setup_constraints();
  velocity.reinit();
  pressure.reinit();
  navier_stokes.setup(true);
  initialize();

  for (unsigned int k = 1; k < time_stepping.get_order(); ++k)
    time_stepping.advance_time();
  
  velocity.solution = velocity.old_old_solution;
  pressure.solution = pressure.old_old_solution;
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
                    1 == 0) ||
                    (time_stepping.get_next_time() == 
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          2 == 0) ||
        (time_stepping.get_next_time() == 
          time_stepping.get_end_time()))
      output();
    
    update_entities();
    time_stepping.advance_time();
  }
  Assert(time_stepping.get_current_time() == velocity_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the velocity function"));
  Assert(time_stepping.get_current_time() == pressure_exact_solution.get_time(),
    ExcMessage("Time mismatch between the time stepping class and the pressure function"));
  velocity_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), prm.flag_spatial_convergence_test);
  pressure_convergence_table.update_table(
    level, time_stepping.get_previous_step_size(), prm.flag_spatial_convergence_test);
  pcout << std::endl;
  pcout << std::endl;
}

template <int dim>
void TGV<dim>::run(const bool flag_convergence_test)
{
  make_grid(prm.initial_refinement_level);
  if (flag_convergence_test)
    for (unsigned int level = prm.initial_refinement_level; 
          level <= prm.final_refinement_level; ++level)
    {
      std::cout.precision(1);
      pcout << "Solving until t = " 
            << std::fixed << time_stepping.get_end_time()
            << " with a refinement level of " << level << std::endl;
      time_stepping.restart();
      solve(level);
      triangulation.refine_global();
    }
  else
  {
    for (unsigned int cycle = 0; 
         cycle < prm.temporal_convergence_cycles; ++cycle)
    {
      double time_step = prm.time_stepping_parameters.initial_time_step *
                         pow(prm.time_step_scaling_factor, cycle);
      std::cout.precision(1);
      pcout << "Solving until t = " 
            << std::fixed << time_stepping.get_end_time()
            << " with a refinement level of " << prm.initial_refinement_level << std::endl;
      time_stepping.restart();
      time_stepping.set_desired_next_step_size(time_step);
      solve(prm.initial_refinement_level);
    }
  }
  
  velocity_convergence_table.print_table();
  pressure_convergence_table.print_table();
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
      simulation.run(parameter_set.flag_spatial_convergence_test);
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
