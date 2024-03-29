#ifndef INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_
#define INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/finite_element_field.h>

#include <fstream>
#include <string>

namespace RMHD
{

using namespace dealii;

template <int dim>
struct ConvergenceAnalysisData
{
  ConvergenceTable                convergence_table;

  const std::shared_ptr<const Entities::FE_FieldBase<dim>> entity;

  const Function<dim>            &exact_solution;

  ConvergenceAnalysisData(const std::shared_ptr<Entities::FE_FieldBase<dim>> &entity,
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
   *
   * @details Test to study the spatial discretization dependence of
   * convergence for a given problem.
   *
   * @note Spatial convergence tests should be performed with a fine
   * time discretization, *i. e.*, a small enough time step.
   */
  spatial = 0x0001,

  /*!
   * @brief Temporal convergence test.
   *
   * @details Test to study the temporal discretization dependence of
   * convergence for a given problem.
   *
   * @note Temporal convergence tests should be performed with a fine
   * spatial discretization, *i. e.*, a triangulation with small enough cells.
   */
  temporal = 0x0002,

  /*!
   * @brief Spatio-temporal convergence test.
   *
   * @details Test to study the dependency on the temporal and the spatial
   * discretization for a given problem.
   *
   * @note Convergence rates are not computed.
   */
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
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ConvergenceTestParameters(const std::string &parameter_filename);

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


/*!
 * @class ConvergenceTestData
 *
 * @brief @ref ConvergenceTestData is a book-keeping class for the errors of a
 * convergence test.
 *
 */
class ConvergenceTestData
{

public:

  ConvergenceTestData(const ConvergenceTestType &type = ConvergenceTestType::temporal);

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant adds
   * the number of DoFs, the cell diameter and the size of the timestep to the
   * convergence table.
   */
  template <int dim, int spacedim>
  void update_table
  (const DoFHandler<dim, spacedim>  &dof_handler,
   const double           time_step,
   const std::map<typename VectorTools::NormType, double> &error_map);

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant adds
   * the number of DoFs and the cell diameter to the convergence table but not the
   * size of the timestep.
   */
  template <int dim, int spacedim>
  void update_table
  (const DoFHandler<dim, spacedim>  &dof_handler,
   const std::map<typename VectorTools::NormType, double> &error_map);

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant adds
   * the size of the timestep but not the number of DoFs and the cell diameter.
   */
  void update_table
  (const double time_step,
   const std::map<typename VectorTools::NormType, double> &error_map);

  /*!
   * @brief Output of the convergence table to a stream object,
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            ConvergenceTestData &data);

  /*!
   * @brief Save results of convergence test to a text file using Org-mode formatting.
   */
  bool save(const std::string &file_name);

private:

  /*!
   * @brief Method which formats the columns of the convergence table.
   *
   * @details The column are printed in scientific notation with a precision of
   * two digits. The convergence rates are also evaluated.
   *
   */
  void format_columns();

  /*!
   * @brief Type of convergence test which is performed.
   *
   * @details The column for computing the convergence rates is selected according
   * to this variable.
   */
  const ConvergenceTestType type;

  /*!
   * @brief Number of rows added to the convergence table..
   */
  unsigned int n_rows;

  /*!
   * @brief Convergence table which stores the error norms and related data.
   */
  ConvergenceTable  table;

  /*!
   * @brief Flag indicating whether the size of the timestep was specified in
   * the last cycle.
   */
  bool step_size_specified{false};

  /*!
   * @brief Flag indicating whether the characteristic cell diameter was specified
   * in the last cycle.
   */
  bool h_max_specified{false};

  /*!
   * @brief Flag indicating whether the L2 error norm was specified in the last
   * cycle.
   */
  bool L2_error_specified{false};

  /*!
   * @brief Flag indicating whether the H1 error norm was specified in the last
   * cycle.
   */
  bool H1_error_specified{false};

  /*!
   * @brief Flag indicating whether the infinity error norm was specified in the
   * last cycle.
   */
  bool Linfty_error_specified{false};

};

}  // namespace ConvergenceTest

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_*/
