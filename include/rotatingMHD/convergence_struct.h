#ifndef INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_
#define INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_


#include <rotatingMHD/entities_structs.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>

#include <fstream>
#include <string>

namespace RMHD
{

using namespace dealii;

template <int dim>
struct ConvergenceAnalysisData
{
  ConvergenceTable                convergence_table;

  const std::string               entity_name;

  const std::shared_ptr<Entities::EntityBase<dim>> entity;

  const Function<dim>             &exact_solution;

  ConvergenceAnalysisData(const std::shared_ptr<Entities::EntityBase<dim>> &entity,
                          const Function<dim>             &exact_solution,
                          const std::string entity_name = "Entity");

  void update_table(const unsigned int  &level,
                    const double        &time_step,
                    const bool          &flag_spatial_convergence);

  void print_table_to_terminal();

  void print_table_to_file(std::string file_name);
};

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_*/