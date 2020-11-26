#ifndef INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_
#define INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_

#include <rotatingMHD/global.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <vector>
#include <map>
#include <memory.h>

namespace RMHD
{

using namespace dealii;

namespace Entities
{

/*!
 * @struct PeriodicBoundaryData
 * @brief A struct encompassing all the information needed to set
 * a periodic boundary condition.
 */
template <int dim>
struct PeriodicBoundaryData
{
  /*!
   * @brief The boundary id pair of the paired edges or faces.
   */
  std::pair<types::boundary_id,types::boundary_id>  boundary_pair;

  /*!
   * @brief The direction in which the edges or faces are parallel.
   */
  unsigned int                                      direction;

  /*!
   * @brief A matrix describing a transformation of the degrees of freedom
   * to be done before pairing. This matrix is only meaningful for vector
   * fields.
   */
  FullMatrix<double>                                rotation_matrix;

  /*!
   * @brief The offset to be done before pairing.
   */
  Tensor<1,dim>                                     offset;

  /*!
   * @brief Constructor.
   */
  PeriodicBoundaryData(const types::boundary_id first_boundary,
                       const types::boundary_id second_boundary,
                       const unsigned int       direction,
                       const FullMatrix<double> rotation_matrix = FullMatrix<double>(IdentityMatrix(dim)),
                       const Tensor<1,dim>      offset = Tensor<1,dim>());
};

/*!
 * @enum BCType
 * @brief An enum class listing the possible types of boundary conditions.
 */
enum class BCType
{
  periodic,
  dirichlet,
  neumann,
  normal_flux,
  tangential_flux
};

/*!
 * @struct BoundaryConditionsBase
 * @brief A structure containing all instances related to the
 * boundary conditions which are independent
 * of the rank of the tensor field.
 */
template <int dim>
struct BoundaryConditionsBase
{
  /*!
   * @brief A typedef for a mapping using boundary ids as keys
   * and shared pointers to functions as values.
   */
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;

  /*!
   * @brief A mapping of boundary ids on which Dirichlet boundary
   * conditions are set and their respective function.
   */
  FunctionMapping                           dirichlet_bcs;

  /*!
   * @brief A vector containing all the @ref PeriodicBoundaryData instances
   * to be applied as boundary conditions.
   */
  std::vector<PeriodicBoundaryData<dim>>    periodic_bcs;

  /*!
   * @brief A multimap of boundary condition types and boundary ids
   * which were mark as having a time-dependent function.
   */
  std::multimap<BCType, types::boundary_id> time_dependent_bcs_map;
};

/*!
 * @struct ScalarBoundaryConditions
 * @brief A structure containing all the instances related to the
 * boundary conditions of a scalar field.
 */
template <int dim>
struct ScalarBoundaryConditions : BoundaryConditionsBase<dim>
{
  /*!
   * @brief A typdef for the a mapping using boundary ids as keys
   * and functions shared pointers as value.
   */
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;
  /*!
   * @brief A mapping of boundary ids on which Neumann boundary
   * conditions are set and their respective function.
   */
  FunctionMapping                           neumann_bcs;

  /*!
   * @brief This method sets a periodic boundary condition by adding a
   * @ref PeriodicBoundaryData object to the member variable
   * @ref BoundaryConditionsBase::periodic_bcs.
   *
   * @details It calls the @ref check_boundary_id method before
   * adding the entry.
   */
  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix = 
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = Tensor<1,dim>());

  /*!
   * @brief This method sets a Dirichlet boundary condition by adding pair of a
   * boundary id and a function to @ref BoundaryConditionsBase::dirichlet_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                         const bool                           time_dependent
                          = false);

  /*!
   * @brief This method sets a Neumann boundary condition by adding a pair of a
   * boundary id and function to @ref BoundaryConditionsBase::dirichlet_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   *
   * @attention The passed function has to match \f$ g(x,t) = \nabla u 
   * \cdot \bs{n} \f$, i.e., a scalar function.
   */
  void set_neumann_bcs(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                         = std::shared_ptr<Function<dim>>(),
                       const bool                           time_depedent
                         = false);

  /*!
   * @brief This method sets the time of the functions by calling their
   * respective methods in a loop.
   */
  void set_time(const double time);

  /*!
   * @brief This method clears all boundary conditions.
   */
  void clear();

  /*!
   * @brief This method copies the content of another @ref ScalarBoundaryConditions
   * instance
   */
  void copy(const ScalarBoundaryConditions<dim> &other);

private:
  /*!
   * @brief A scalar zero function used for homogeneous boundary
   * conditions.
   */
  const std::shared_ptr<Function<dim>>  zero_function_ptr
    = std::make_shared<Functions::ZeroFunction<dim>>();

  /*!
   * @brief This method checks if the passed boundary id is already constrained.
   *
   * @details It returns an error if the passed boundary id is constrained.
   */
  void check_boundary_id(const types::boundary_id boundary_id) const;

};

/*!
 * @struct VectorBoundaryConditions
 * @brief A structure containing all the instances related to the
 * boundary conditions of a vector field.
 */
template <int dim>
struct VectorBoundaryConditions : BoundaryConditionsBase<dim>
{
  /*!
   * @brief A typedef for the a mapping using boundary ids as keys
   * and shared pointers to functions as values.
   */
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;

  /*!
   * @brief A typdef for the a mapping using boundary ids as keys
   * and tensor functions shared pointers as value.
   */
  using TensorFunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<TensorFunction<1,dim>>>;
  /*!
   * @brief A mapping of boundary ids on which Neumann boundary
   * conditions are set and their respective function.
   */
  TensorFunctionMapping   neumann_bcs;

  /*!
   * @brief A mapping of boundary ids on which normal flux boundary
   * conditions are set and their respective functions.
   */
  FunctionMapping         normal_flux_bcs;

  /*!
   * @brief A mapping of boundary ids on which tangential flux boundary
   * conditions are set and their respective functions.
   */
  FunctionMapping         tangential_flux_bcs;

  /*!
   * @brief This method sets a periodic boundary condition by adding a
   * @ref PeriodicBoundaryData to @ref BoundaryConditionsBase::periodic_bcs.
   *
   * @details It calls the @ref check_boundary_id method before
   * adding the entry.
   */
  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix
                          = FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset
                          = Tensor<1,dim>());

  /*!
   * @brief This method sets a Dirichlet boundary conditions by adding a
   * boundary id and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                         const bool                           time_depedent
                          = false);

  /*!
   * @brief This methods sets a Neumann boundary conditions by adding a boundary
   * id and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time-dependent according to the boolean passed.
   *
   * @attention The passed function has to match \f$ \bs{g}(x,t) = \bs{T} 
   * \cdot \bs{n} \f$, i.e., a vector function.
   */
  void set_neumann_bcs(const types::boundary_id                       boundary_id,
                       const std::shared_ptr<TensorFunction<1, dim>>  &function
                          = std::shared_ptr<TensorFunction<1, dim>>(),
                       const bool                                     time_depedent
                          = false);

  /*!
   * @brief This method sets a normal flux boundary condition by adding a
   * boundary id and function pair to @ref normal_flux_bcs.
   *
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_normal_flux_bcs(const types::boundary_id             boundary_id,
                           const std::shared_ptr<Function<dim>> &function
                              = std::shared_ptr<Function<dim>>(),
                           const bool                           time_depedent
                              = false);

  /*!
   * @brief This method sets a tangential flux boundary condition by adding
   * a boundary id and function pair to @ref tangential_flux_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_tangential_flux_bcs(const types::boundary_id             boundary_id,
                               const std::shared_ptr<Function<dim>> &function
                                  = std::shared_ptr<Function<dim>>(),
                               const bool                           time_depedent
                                  = false);

  /*!
   * @brief This method sets the time of the functions by calling their
   * respective methods in a loop.
   */
  void set_time(const double time);

  /*!
   * @brief This method clears all boundary conditions.
   */
  void clear();

  /*!
   * @brief This method copies the content of another @ref VectorBoundaryConditions
   * instance.
   */
  void copy(const VectorBoundaryConditions<dim> &other);
  
private:

  /*!
   * @brief A vector zero function used for homogeneous boundary
   * conditions.
   */
  const std::shared_ptr<Function<dim>>  zero_function_ptr
    = std::make_shared<Functions::ZeroFunction<dim>>(dim);

  /*!
   * @brief A scalar zero tensor function used for homogeneous boundary
   * conditions.
   */
  std::shared_ptr<TensorFunction<1, dim>>  zero_vector = 
                      std::make_shared<ZeroTensorFunction<1,dim>>();


  /*!
   * @brief Checks if the passed boundary id was already constrained.
   *
   * @details It returns an error if it was.
   */
  void check_boundary_id(const types::boundary_id boundary_id) const;

};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_ */
