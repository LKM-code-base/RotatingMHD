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

namespace ConvergenceTest
{

/*!
 * @brief Enumeration for convergence test type.
 */
enum class ConvergenceTestType
{
  /*!
   * @brief Spatial convergence test.
   * @details Test to study the spatial discretization dependence of
   * convergence for a given problem.
   * @note Spatial convergence tests should be performed with a fine
   * time discretization, *i. e.*, a small enough time step.
   */
  spatial = 0x0001,

  /*!
   * @brief Temporal convergence test.
   * @details Test to study the temporal discretization dependence of
   * convergence for a given problem.
   * @note Temporal convergence tests should be performed with a fine
   * spatial discretization, *i. e.*, a triangulation with small enough cells.
   */
  temporal = 0x0002,

  spatio_temporal = spatial|temporal
};

/*!
 * @struct ConvergenceTestParameters
 *
 * @brief @ref ConvergenceTestParameters contains parameters which are
 * related to convergence tests.
 */
struct ConvergenceTestParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvergenceTestParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const ConvergenceTestParameters &prm);

  /*!
   * @brief The type of convergence test (spatial or temporal).
   */
  ConvergenceTestType test_type;

  /*!
   * Number of spatial convergence cycles.
   */
  unsigned int        n_spatial_cycles;

  /*!
   * @brief Factor \f$ s \f$ of the reduction of the timestep between two
   * subsequent levels, *i. e.*, \f$ \Delta t_{l+1} = s \Delta t_l\f$.
   *
   * @details The factor \f$ s \f$ must be positive and less than unity.
   */
  double              step_size_reduction_factor;

  /*!
   * @brief Number of temporal convergence cycles.
   */
  unsigned int        n_temporal_cycles;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvergenceTestParameters &prm);

}  // namespace ConvergenceTest

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_*/
