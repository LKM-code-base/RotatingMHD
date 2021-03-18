
#ifndef INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_
#define INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_

#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/boundary_conditions.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>

#include <vector>
#include <map>
#include <memory>
#include <set>

namespace RMHD
{

using namespace dealii;
/*!
 * @namespace Entities
 * @brief The namescape encompasses the numerical representation
 * of scalar and vector fields.
 */
namespace Entities
{

/*!
 * @struct EntityBase
 * @brief This struct gathers all the numerical attributes that are
 * independent of the rank of the tensor.
 */

template <int dim>
struct EntityBase
{
public:
  /*!
   * @brief Constructor.
   */
  EntityBase(const unsigned int                               n_components,
             const unsigned int                               fe_degree,
             const parallel::distributed::Triangulation<dim> &triangulation,
             const std::string                               &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  EntityBase(const EntityBase<dim>  &entity,
             const std::string      &new_name = "entity");

  /*!
   * @brief Number of vector components.
   */
  const unsigned int                n_components;

  /*!
   * @brief The degree of the finite element.
   */
  const unsigned int                fe_degree;

  /*!
   * @brief The MPI communicator.
   */
  const MPI_Comm                    mpi_communicator;

  /*!
   * @brief The DoFHandler<dim> instance of the entity.
   */
  std::shared_ptr<DoFHandler<dim>>  dof_handler;

  /*!
   * @brief Name of the physical field which is contained in the entity.
   */
  const std::string                 name;

  /*!
   * @brief The AffineConstraints<double> instance handling the
   * hanging nodes.
   */
  AffineConstraints<double>         hanging_nodes;

  /*!
   * @brief The AffineConstraints<double> instance handling the
   * hanging nodes and the boundary conditions.
   */
  AffineConstraints<double>         constraints;

  /*!
   * @brief The set of the degrees of freedom owned by the processor.
   */
  IndexSet                          locally_owned_dofs;

  /*!
   * @brief The set of the degrees of freedom that are relevant for
   * the processor.
   */
  IndexSet                          locally_relevant_dofs;

  /*!
   * @brief Vector containing the solution at the current time.
   */
  LinearAlgebra::MPI::Vector        solution;

  /*!
   * @brief Vector containing the solution one time step prior to the
   * current time.
   */
  LinearAlgebra::MPI::Vector        old_solution;

  /*!
   * @brief Vector containing the solution two time step prior to the
   * current time.
   */
  LinearAlgebra::MPI::Vector        old_old_solution;

  /*!
   * @brief The entity's distributed vector.
   *
   * @details It is used to initiate the right hand sides of the
   * linear systems and the distributed instances of the
   * solution vectors needed to perform algebraic operations with them.
   */
  LinearAlgebra::MPI::Vector        distributed_vector;

  /*!
   * @brief Method returning a reference to the triangulation.
   */
  const parallel::distributed::Triangulation<dim> &get_triangulation() const;

  /*!
   * @brief Initializes the solution vectors by calling their respective
   * reinit method.
   */
  void reinit();

  /*!
   * @brief Passes the contents of a solution vector to the one prior to it.
   */
  void update_solution_vectors();

  /*!
   * @brief Sets all entries of all solution vectors to zero.
   */
  void set_solution_vectors_to_zero();

  /*!
   * @brief Empty virtual method introduced to gather @ref ScalarEntity
   * and @ref VectorEntity in a vector and call
   * @ref ScalarEntity::setup_dofs and @ref VectorEntity::setup_dofs
   * respectively.
   */
  virtual void setup_dofs() = 0;

  /*!
   * @brief Empty virtual method introduced to gather @ref ScalarEntity
   * and @ref VectorEntity in a vector and call
   * @ref ScalarEntity::apply_boundary_conditions and
   * @ref VectorEntity::apply_boundary_conditions respectively.
   */
  virtual void apply_boundary_conditions() = 0;

  /*!
   * @brief Empty virtual method introduced to gather @ref ScalarEntity
   * and @ref VectorEntity as EntityBase instances and call
   * @ref ScalarEntity::close_boundary_conditions and
   * @ref VectorEntity::close_boundary_conditions respectively.
   */
  virtual void close_boundary_conditions() = 0;

  /*!
   * @brief Empty virtual method introduced to gather @ref ScalarEntity
   * and @ref VectorEntity in a vector and call
   * @ref ScalarEntity::update_boundary_conditions and
   * @ref VectorEntity::update_boundary_conditions respectively.
   */
  virtual void update_boundary_conditions() = 0;

  /*!
   * @brief Empty virtual method introduced to gather @ref ScalarEntity
   * and @ref VectorEntity as EntityBase instances and call
   * @ref ScalarEntity::clear_boundary_conditions and
   * @ref VectorEntity::clear_boundary_conditions respectively.
   */
  virtual void clear_boundary_conditions() = 0;

  /*!
   * @brief Returns the value of @ref flag_child_entity.
   */
  bool is_child_entity() const;

protected:
  /*!
   * @brief A flag indicating whether the entity is a child entity. This menas
   * that the entity was instantiated using the copy constructor.
   *
   * @details This flag is used to avoid a double distribution of the degrees of
   * freedom.
   */
  const bool  flag_child_entity;

  /*!
   * @brief A flag indicating whether @ref setup_dofs was called.
   */
  bool        flag_setup_dofs;

private:
  /*!
   * @brief Reference to the underlying triangulation.
   */
  const parallel::distributed::Triangulation<dim> &triangulation;
};



template <int dim>
inline bool EntityBase<dim>::is_child_entity() const
{
  return (flag_child_entity);
}



template <int dim>
inline const parallel::distributed::Triangulation<dim> &EntityBase<dim>::get_triangulation() const
{
  return (triangulation);
}

  /*!
   * @struct VectorEntity
   * @brief Numerical representation of a vector field.
   */
template <int dim>
struct VectorEntity : EntityBase<dim>
{
  /*!
   * @brief Constructor.
   */
  VectorEntity(const unsigned int                               fe_degree,
               const parallel::distributed::Triangulation<dim> &triangulation,
               const std::string                               &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  VectorEntity(const VectorEntity<dim>  &entity,
               const std::string        &new_name);

  /*!
   * @brief The finite element of the vector field.
   */
  FESystem<dim>                 fe;

  /*!
   * @brief @ref VectorBoundaryConditions instance bookkeeping the
   * boundary conditions of the vector field.
   */
  VectorBoundaryConditions<dim> boundary_conditions;

  /*!
   * @brief Set ups the degrees of freedom of the vector field.
   *
   * @details It distributes the degrees of freedom bases on @ref fe;
   * extracts the @ref locally_owned_dofs and the @ref locally_relevant_dofs;
   * and makes the hanging node constraints contained in @ref hanging_nodes.
   */
  virtual void setup_dofs() override;

  /*!
   * @brief Applies all specified boundary conditions to the @ref constraints
   * of the vector field.
   *
   * @details It loops over the elements stored in @ref boundary_conditions
   * and modifies @ref constraints accordingly.
   *
   * @attention This method has to be called even if no boundary conditions
   * are applied as the method initiates @ref constraints, which is used
   * througout the solver.
   */
  virtual void apply_boundary_conditions() override;

  /*!
   * @brief Closes the @ref boundary_conditions and prints a summary
   * of the boundary conditions to the terminal.
   */
  virtual void close_boundary_conditions() override;

  /*!
   * @brief Updates the time dependent boundary conditions.
   *
   * @details It loops over all boundary condition marked as time
   * dependent and reapplies the constraints into a temporary
   * AffineConstraints<double> instance which is then merge into @ref
   * constraints.
   *
   * @attention Make sure to advance the underlying function in time
   * using the @ref VectorBoundaryConditions::set_time method before
   * calling this method. Otherwise the method will just reapply the
   * same boundary conditions.
   */
  virtual void update_boundary_conditions() override;

  /*!
   * @brief Clears the @ref boundary_conditions and the @ref constraints.
   */
  virtual void clear_boundary_conditions() override;

  /*!
   * @brief This method evaluates the value of the continous vector
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  Tensor<1,dim> point_value(const Point<dim>  &point) const;
};

  /*!
   * @struct VectorEntity
   * @brief Numerical representation of a scalar field.
   */
template <int dim>
struct ScalarEntity : EntityBase<dim>
{
  /*!
   * @brief Constructor.
   */
  ScalarEntity(const unsigned int                               fe_degree,
               const parallel::distributed::Triangulation<dim> &triangulation,
               const std::string                               &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  ScalarEntity(const ScalarEntity<dim>  &entity,
               const std::string        &new_name = "entity");

  /*!
   * @brief The finite element of the scalar field.
   */
  FE_Q<dim> fe;

  /*!
   * @brief @ref ScalarBoundaryConditions instance bookkeeping the
   * boundary conditions of the scalar field.
   */
  ScalarBoundaryConditions<dim>       boundary_conditions;

  /*!
   * @brief Set ups the degrees of freedom of the scalar field.
   * @details It distributes the degrees of freedom bases on @ref fe;
   * extracts the @ref locally_owned_dofs and the @ref locally_relevant_dofs;
   * and makes the hanging node constraints contained in @ref hanging_nodes.
   */
  virtual void setup_dofs() override;

  /*!
   * @brief Applies all the boundary conditions into the @ref constraints
   * of the scalar field.
   *
   * @details It loops over the elements stored in @ref boundary_conditions
   * and modifies @ref constraints accordingly.
   *
   * @attention This method has to be called even if no boundary conditions
   * are applied because the method initiates @ref constraints, which are used
   * througout the solver.
   */
  virtual void apply_boundary_conditions() override;

  /*!
   * @brief Closes the @ref boundary_conditions and prints a summary
   * of the boundary conditions to the terminal.
   */
  virtual void close_boundary_conditions() override;

  /*!
   * @brief Updates the time dependent boundary conditions.
   *
   * @details It loops over all boundary condition marked as time
   * dependent and reapplies the constraints into a temporary
   * AffineConstraints<double> instance which is then merge into @ref
   * constraints.
   *
   * @attention Make sure to advance the underlying function in time
   * using the @ref ScalarBoundaryConditions::set_time method before
   * calling this method. Otherwise the method will just re-apply the
   * same boundary conditions.
   */
  virtual void update_boundary_conditions() override;

  /*!
   * @brief Clears the @ref boundary_conditions and the @ref constraints.
   */
  virtual void clear_boundary_conditions() override;

  /*!
   * @brief This method evaluates the value of the continous scalar
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  double point_value(const Point<dim> &point) const;
};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_ENTITIES_STRUCTS_H_ */
