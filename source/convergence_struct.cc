#include <rotatingMHD/convergence_struct.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

using namespace dealii;

template <int dim>
ConvergenceAnalysisData<dim>::ConvergenceAnalysisData(
  const std::shared_ptr<Entities::EntityBase<dim>> &entity,
  const Function<dim>             &exact_solution,
  const std::string               entity_name)
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
  convergence_table.declare_column("H1");
  convergence_table.declare_column("Linfty");
  convergence_table.set_scientific("dt", true);
  convergence_table.set_scientific("hmax", true);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);
  convergence_table.set_precision("dt", 2);
  convergence_table.set_precision("hmax", 2);
  convergence_table.set_precision("L2", 6);
  convergence_table.set_precision("H1", 6);
  convergence_table.set_precision("Linfty", 6);
}

template <int dim>
void ConvergenceAnalysisData<dim>::update_table(
  const unsigned int  &level,
  const double        &time_step,
  const bool          &flag_spatial_convergence)
{
  Vector<double> cellwise_difference(
    entity->get_triangulation().n_active_cells());

  QGauss<dim>    quadrature_formula(entity->fe_degree + 2);
  const QTrapez<1>     trapezoidal_rule;
  const QIterated<dim> iterated_quadrature_rule(trapezoidal_rule,
                                                entity->fe_degree * 2 + 1);
  
  VectorTools::integrate_difference(*entity->dof_handler,
                                    entity->solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::L2_norm);
  
  const double L2_error =
    VectorTools::compute_global_error(entity->get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::L2_norm);

  VectorTools::integrate_difference(*entity->dof_handler,
                                    entity->solution,
                                    exact_solution,
                                    cellwise_difference,
                                    quadrature_formula,
                                    VectorTools::H1_norm);
  
  const double H1_error =
    VectorTools::compute_global_error(entity->get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::H1_norm);

  VectorTools::integrate_difference(*entity->dof_handler,
                                    entity->solution,
                                    exact_solution,
                                    cellwise_difference,
                                    iterated_quadrature_rule,
                                    VectorTools::Linfty_norm);
  
  const double Linfty_error =
    VectorTools::compute_global_error(entity->get_triangulation(),
                                      cellwise_difference,
                                      VectorTools::Linfty_norm);

  convergence_table.add_value("level", level);
  convergence_table.add_value("dt", time_step);
  convergence_table.add_value("cells", entity->get_triangulation().n_global_active_cells());
  convergence_table.add_value("dofs", (entity->dof_handler)->n_dofs());
  convergence_table.add_value("hmax", GridTools::maximal_cell_diameter(entity->get_triangulation()));
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

  std::string reference_column = (flag_spatial_convergence) ? 
                                  "hmax" : "dt";

  convergence_table.evaluate_convergence_rates(
                              "L2",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  convergence_table.evaluate_convergence_rates(
                              "H1",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
  convergence_table.evaluate_convergence_rates(
                              "Linfty",
                              reference_column,
                              ConvergenceTable::reduction_rate_log2,
                              1);
}

template <int dim>
void ConvergenceAnalysisData<dim>::print_table_to_terminal()
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
  std::cout << std::endl;
  std::cout 
    << "                               " << entity_name 
    << " convergence table" << std::endl
    << "==============================================="
    << "===============================================" 
    << std::endl;
  convergence_table.write_text(std::cout);
  }
}

template <int dim>
void ConvergenceAnalysisData<dim>::print_table_to_file(
  std::string file_name)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    file_name += ".txt";

    std::ofstream file(file_name);

    convergence_table.write_text(
      file,
      TableHandler::TextOutputFormat::org_mode_table);
  }
}


} // namespace RMHD

template struct RMHD::ConvergenceAnalysisData<2>;
template struct RMHD::ConvergenceAnalysisData<3>;