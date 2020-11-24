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

  const std::shared_ptr<const Entities::EntityBase<dim>> entity;

  const Function<dim>            &exact_solution;

  ConvergenceAnalysisData(const std::shared_ptr<Entities::EntityBase<dim>> &entity,
                          const Function<dim>             &exact_solution);

  void update_table(const unsigned int  level,
                    const double        time_step,
                    const bool          flag_spatial_convergence);

  /*!
   * @brief Output of the convergence table to a stream object,
   */
  template<typename Stream, int dimension>
  friend Stream& operator<<(Stream &stream,
                            const ConvergenceAnalysisData<dimension> &data);

  void write_text(std::string filename) const;

};

template<typename Stream, int dim>
Stream& operator<<(Stream &stream,
                   const ConvergenceAnalysisData<dim> &data);

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_*/
