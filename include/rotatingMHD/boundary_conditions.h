#ifndef INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_
#define INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_

#include <rotatingMHD/global.h>

#include <deal.II/base/function.h>

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
   * @brief The rotation to be done before pairing.
   */
  FullMatrix<double>                                rotation_matrix;

  /*!
   * @brief The offset to be done before pairing.
   */
  Tensor<1,dim>                                     offset;

  /*!
   * @brief Constructor.
   */
  PeriodicBoundaryData(types::boundary_id first_boundary,
                       types::boundary_id second_boundary,
                       unsigned int       direction,
                       FullMatrix<double> rotation_matrix,
                       Tensor<1,dim>      offset)
  : boundary_pair(std::make_pair(first_boundary, second_boundary)),
    direction(direction),
    rotation_matrix(rotation_matrix),
    offset(offset)
  {}
};

/*!
 * @enum BCType
 * @brief An enum class listing the possible boundary types.
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
 * @brief An struct containing all the instances related to the 
 * boundary conditions which are independent
 * of the rank of the tensor field.
 */
template <int dim>
struct BoundaryConditionsBase
{
  /*!
   * @brief A typdef for the a mapping using boundary ids as keys
   * and functions shared pointers as value.
   */
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;

  /*!
   * @brief A mapping of boundary ids on which Dirichlet boundary
   * conditions are set and their respective function.
   */
  FunctionMapping                           dirichlet_bcs;

  /*!
   * @brief A mapping of boundary ids on which Neumann boundary
   * conditions are set and their respective function.
   */
  FunctionMapping                           neumann_bcs;

  /*!
   * @brief A vector containing all the @ref PeriodicBoundaryData instances
   * to be applied as boundary condition.
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
 * @brief An struct containing all the instances related to the 
 * boundary conditions of a scalar field.
 */
template <int dim>
struct ScalarBoundaryConditions : BoundaryConditionsBase<dim>
{
  /*!
   * @brief It sets a periodic boundary conditions by adding a 
   * @ref PeriodicBoundaryData into @ref BoundaryConditionsBase::periodic_bcs.
   * @details It calls the @ref check_boundary_id method before
   * adding the entry.
   */
  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix = 
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = 
                                Tensor<1,dim>());

  /*!
   * @brief Sets a Dirichlet boundary conditions by adding a boundary id
   * and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                         const bool                           time_dependent
                          = false);

  /*!
   * @brief Sets a Neumann boundary conditions by adding a boundary id
   * and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   * @attention The passed function has to match \f$ g(x,t) = \nabla u 
   * \cdot \bs{n} \f$, i.e. a scalar function.
   */
  void set_neumann_bcs(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                       const bool                           time_depedent
                          = false);

  /*!
   * @brief Sets the time of the functions by calling their respective
   * set_time method in a loop.
   */
  void set_time(const double time);

  /*!
   * @brief Clears all the boundary conditions.
   */
  void clear();

private:

  /*!
   * @brief Checks if the passed boundary id was already constrained.
   * @details It returns an error if it was.
   */
  void check_boundary_id(types::boundary_id boundary_id) const;

};

/*!
 * @struct ScalarBoundaryConditions
 * @brief An struct containing all the instances related to the 
 * boundary conditions of a vector field.
 */
template <int dim>
struct VectorBoundaryConditions : BoundaryConditionsBase<dim>
{
    /*!
   * @brief A typdef for the a mapping using boundary ids as keys
   * and functions shared pointers as value.
   */
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;
  /*!
   * @brief A mapping of boundary ids on which normal flux boundary
   * conditions are set and their respective function.
   */
  FunctionMapping   normal_flux_bcs;

  /*!
   * @brief A mapping of boundary ids on which tangential flux boundary
   * conditions are set and their respective function.
   */
  FunctionMapping   tangential_flux_bcs;

  /*!
   * @brief It sets a periodic boundary conditions by adding a 
   * @ref PeriodicBoundaryData into @ref BoundaryConditionsBase::periodic_bcs.
   * @details It calls the @ref check_boundary_id method before
   * adding the entry.
   */
  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix = 
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = 
                                Tensor<1,dim>());

  /*!
   * @brief Sets a Dirichlet boundary conditions by adding a boundary id
   * and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                         const bool                           time_depedent
                          = false);

  /*!
   * @brief Sets a Neumann boundary conditions by adding a boundary id
   * and function pair to @ref BoundaryConditionsBase::dirichlet_bcs.
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   * @attention The passed function has to match \f$ \bs{g}(x,t) = \bs{T} 
   * \cdot \bs{n} \f$, i.e. a vector function.
   */
  void set_neumann_bcs(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                       const bool                           time_depedent
                          = false);

  /*!
   * @brief Sets a normal flux boundary conditions by adding a boundary id
   * and function pair to @ref normal_flux_bcs.
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
   * @brief Sets a tangential flux boundary conditions by adding a boundary id
   * and function pair to @ref tangential_flux_bcs.
   * @details If no function is explicitly passed, it assumes that the 
   * degrees of freedom are homogenously constrained. It calls the 
   * @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_tangential_flux_bcs(const types::boundary_id             boundary_id,
                               const std::shared_ptr<Function<dim>> &function
                                  = std::shared_ptr<Function<dim>>(),
                               const bool                           time_depedent
                                  = false);

  /*!
   * @brief Sets the time of the functions by calling their respective
   * set_time method in a loop.
   */
  void set_time(const double time);

  /*!
   * @brief Clears all the boundary conditions.
   */
  void clear();

private:

  /*!
   * @brief Checks if the passed boundary id was already constrained.
   * @details It returns an error if it was.
   */
  void check_boundary_id(types::boundary_id boundary_id) const;

};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_ */