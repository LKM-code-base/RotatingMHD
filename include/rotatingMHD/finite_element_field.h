#ifndef INCLUDE_ROTATINGMHD_FINITE_ELEMENT_FIELD_H_
#define INCLUDE_ROTATINGMHD_FINITE_ELEMENT_FIELD_H_

#include <rotatingMHD/global.h>
#include <rotatingMHD/boundary_conditions.h>

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>

#include <vector>
#include <map>
#include <memory>
#include <set>

namespace RMHD
{

using namespace dealii;

/*!
 * @namespace Entities
 *
 * @brief The namescape encompasses the numerical representation
 * of scalar and vector fields.
 */
namespace Entities
{

/*!
 * @struct FE_FieldBase
 *
 * @brief This struct gathers all the numerical attributes that are
 * independent of the rank of the tensor.
 */
template <int dim, typename VectorType = LinearAlgebra::MPI::Vector>
class FE_FieldBase
{
public:
  using value_type = typename VectorType::value_type;

  /*!
   * @brief Constructor.
   */
  FE_FieldBase(const Triangulation<dim>  &triangulation,
               const std::string         &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  FE_FieldBase(const FE_FieldBase<dim, VectorType>  &entity,
               const std::string        &new_name = "entity");

  /*!
   * @brief Returns a const reference to the @ref dof_handler
   */
  const DoFHandler<dim>&  get_dof_handler() const;

  /*!
   * @brief Returns a const reference to the @ref finite_element
   */
  const FiniteElement<dim>&  get_finite_element() const;

  /*!
   * @brief Returns a const reference to the @ref locally_owned_dofs
   */
  const IndexSet& get_locally_owned_dofs() const;

  /*!
   * @brief Returns a const reference to the @ref locally_relevant_owned_dofs
   */
  const IndexSet& get_locally_relevant_dofs() const;

  /*!
   * @brief Returns a const reference to the @ref constraints
   */
  const AffineConstraints<value_type>&  get_constraints() const;

  /*!
   * @brief Returns a const reference to the @ref hanging_node_constraints
   */
  const AffineConstraints<value_type>&  get_hanging_node_constraints() const;

  const BoundaryConditionsBase<dim> &
  get_boundary_conditions() const;

  BoundaryConditionsBase<dim> & get_boundary_conditions();


  const typename BoundaryConditionsBase<dim>::BCMapping &
  get_dirichlet_boundary_conditions() const;

  /*!
   * @brief Set a Dirichlet boundary condition.
   */
  void set_dirichlet_boundary_condition(const types::boundary_id ,
                                        const std::shared_ptr<Function<dim>> &function
                                          = std::shared_ptr<Function<dim>>(),
                                        const bool time_dependent_bc = false);

  /*!
   * @brief Set a periodic boundary condition.
   */
  void set_periodic_boundary_condition(const types::boundary_id  first_bndry_id,
                                       const types::boundary_id  second_bndry_id,
                                       const unsigned int        direction);

  /*!
   * @brief Returns the number of degrees of freedom.
   */
  types::global_dof_index n_dofs() const;

  /*!
   * @brief Returns the number of components of the finite element.
   */
  unsigned int n_components() const;

  /*!
   * @brief Returns the polynomial degree of the finite element.
   */
  unsigned int fe_degree() const;

  /*!
   * @brief Name of the physical field which is contained in the entity.
   */
  const std::string                 name;

  /*!
   * @brief Vector containing the solution at the current time.
   */
  VectorType  solution;

  /*!
   * @brief Vector containing the solution one time step prior to the
   * current time.
   */
  VectorType  old_solution;

  /*!
   * @brief Vector containing the solution two time step prior to the
   * current time.
   */
  VectorType  old_old_solution;

  /*!
   * @brief The entity's distributed vector.
   *
   * @details It is used to initiate the right hand sides of the
   * linear systems and the distributed instances of the
   * solution vectors needed to perform algebraic operations with them.
   */
  VectorType  distributed_vector;

  /*!
   * @brief Method returning a reference to the triangulation.
   */
  const Triangulation<dim> &get_triangulation() const;

  /*!
   * @details Release all memory and return all objects to a state just like
   * after having called the default constructor.
   */
  virtual void clear();

  /*!
   * @brief Initializes the solution vectors by calling their respective
   * reinit method.
   */
  void setup_vectors();

  /*!
   * @brief Passes the contents of a solution vector to the one prior to it.
   */
  virtual void update_solution_vectors();

  /*!
   * @brief Sets all entries of all solution vectors to zero.
   */
  void set_solution_vectors_to_zero();

  /*!
   * @brief Virtual method introduced to gather @ref FE_ScalarField
   * and @ref FE_VectorField in a vector and call
   * @ref FE_ScalarField::setup_dofs and @ref FE_VectorField::setup_dofs
   * respectively.
   */
  virtual void setup_dofs();

  /*!
   * @brief Virtual method introduced to gather @ref FE_ScalarField
   * and @ref FE_VectorField and call
   * @ref FE_ScalarField::apply_boundary_conditions and
   * @ref FE_VectorField::apply_boundary_conditions respectively.
   */
  virtual void apply_boundary_conditions(const bool check_regularity = true);

  /*!
   * @brief Closes the @ref boundary_conditions and prints a summary
   * of the boundary conditions to the terminal.
   */
  void close_boundary_conditions(const bool print_summary = true);

  void setup_boundary_conditions();

  /*!
   * @brief Virtual method introduced to gather @ref FE_ScalarField
   * and @ref FE_VectorField and call
   * @ref FE_ScalarField::update_boundary_conditions and
   * @ref FE_VectorField::update_boundary_conditions respectively.
   */
  virtual void update_boundary_conditions();

  /*!
   * @brief Clears the @ref boundary_conditions and the @ref constraints.
   */
  void clear_boundary_conditions();

  /*!
   * @brief Returns the value of @ref flag_child_entity.
   */
  bool is_child_entity() const;

  /*!
   * @brief Computes the error of the discrete solution w.r.t. to the exact solution
   * specified by the function @p exact_solution. The error is computed in the L2 norm, the H1 norm
   * and the infinity norm and return as mapping.
   */
  std::map<typename VectorTools::NormType, double> compute_error
  (const Function<dim>  &exact_solution,
   const Mapping<dim>   &external_mapping) const;

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

  /*!
   * @brief The MPI communicator.
   */
//  const MPI_Comm    mpi_communicator;

  /*!
   * @brief Reference to the underlying triangulation.
   */
  const Triangulation<dim> &triangulation;

  /*!
   * @brief The DoFHandler<dim> instance of the entity.
   */
  std::shared_ptr<DoFHandler<dim>>    dof_handler;

  /*!
   * @brief Finite element.
   */
  std::shared_ptr<FiniteElement<dim>> finite_element;

  /*!
   * @brief @ref BoundaryConditions instance for bookkeeping the
   * boundary conditions of the finite element field.
   */
  BoundaryConditionsBase<dim>*        boundary_conditions;

  /*!
   * @brief The AffineConstraints<double> instance handling the
   * hanging nodes.
   */
  AffineConstraints<value_type>       hanging_node_constraints;

  /*!
   * @brief The AffineConstraints<double> instance handling the
   * hanging nodes and the boundary conditions.
   */
  AffineConstraints<value_type>       constraints;

  /*!
   * @brief The set of the degrees of freedom owned by the processor.
   */
  IndexSet                            locally_owned_dofs;

  /*!
   * @brief The set of the degrees of freedom that are relevant for
   * the processor.
   */
  IndexSet                            locally_relevant_dofs;

private:
  /*!
   * @brief Method applying periodic boundary conditions to the @ref constraints.
   */
  void apply_periodicity_constraints();

  /*!
   * @brief Method applying periodic boundary conditions to the @ref constraints.
   */
  void apply_dirichlet_constraints();

protected:
  template<typename T>
  void send_point_data(T                 &point_value,
                       const Point<dim>  &point,
                       const bool         point_found) const;

  template<typename T>
  void send_point_data_vector(std::vector<T>    &point_value,
                              const Point<dim>  &point,
                              const bool         point_found) const;
};

template <int dim, typename VectorType>
inline const DoFHandler<dim> &
FE_FieldBase<dim, VectorType>::get_dof_handler() const
{
  return (*dof_handler);
}

template <int dim, typename VectorType>
inline const FiniteElement<dim> &
FE_FieldBase<dim, VectorType>::get_finite_element() const
{
  return (*finite_element);
}

template <int dim, typename VectorType>
inline const IndexSet &
FE_FieldBase<dim, VectorType>::get_locally_owned_dofs() const
{
  return (locally_owned_dofs);
}

template <int dim, typename VectorType>
inline const IndexSet &
FE_FieldBase<dim, VectorType>::get_locally_relevant_dofs() const
{
  return (locally_relevant_dofs);
}

template <int dim, typename VectorType>
inline const AffineConstraints<typename FE_FieldBase<dim, VectorType>::value_type> &
FE_FieldBase<dim, VectorType>::get_constraints() const
{
  return (constraints);
}

template <int dim, typename VectorType>
inline BoundaryConditionsBase<dim> &
FE_FieldBase<dim, VectorType>::get_boundary_conditions()
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));
  return (*boundary_conditions);
}

template <int dim, typename VectorType>
inline const BoundaryConditionsBase<dim> &
FE_FieldBase<dim, VectorType>::get_boundary_conditions() const
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));
  return (*boundary_conditions);
}

template <int dim, typename VectorType>
inline const typename BoundaryConditionsBase<dim>::BCMapping &
FE_FieldBase<dim, VectorType>::get_dirichlet_boundary_conditions() const
{
  return (boundary_conditions->dirichlet_bcs);
}

template <int dim, typename VectorType>
inline const AffineConstraints<typename FE_FieldBase<dim, VectorType>::value_type> &
FE_FieldBase<dim, VectorType>::get_hanging_node_constraints() const
{
  return (hanging_node_constraints);
}

template <int dim, typename VectorType>
inline bool FE_FieldBase<dim, VectorType>::is_child_entity() const
{
  return (flag_child_entity);
}

template <int dim, typename VectorType>
inline types::global_dof_index
FE_FieldBase<dim, VectorType>::n_dofs() const
{
  return (dof_handler->n_dofs());
}

template <int dim, typename VectorType>
inline unsigned int
FE_FieldBase<dim, VectorType>::n_components() const
{
  return (finite_element->n_components());
}

template <int dim, typename VectorType>
inline unsigned int
FE_FieldBase<dim, VectorType>::fe_degree() const
{
  return (finite_element->degree);
}

template <int dim, typename VectorType>
inline const Triangulation<dim> &
FE_FieldBase<dim, VectorType>::get_triangulation() const
{
  return (triangulation);
}

template <int dim, typename VectorType>
inline void FE_FieldBase<dim, VectorType>::setup_boundary_conditions()
{
  Assert(this->boundary_conditions != nullptr,
         ExcMessage("Boundary conditions object is not initialized."));

  this->boundary_conditions->extract_boundary_ids();
}


/*!
 * @struct FE_VectorField
 * @brief Numerical representation of a vector field.
 */
template <int dim, typename VectorType = LinearAlgebra::MPI::Vector>
class FE_VectorField: public FE_FieldBase<dim, VectorType>
{
public:
  /*!
   * @brief Constructor.
   */
  FE_VectorField(const unsigned int                               fe_degree,
                 const Triangulation<dim> &triangulation,
                 const std::string                               &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  FE_VectorField(const FE_VectorField<dim, VectorType> &entity,
                 const std::string                     &new_name);

  const typename VectorBoundaryConditions<dim>::NeumannBCMapping &
  get_neumann_boundary_conditions() const;

  /*!
   * @brief Set a Neumann boundary condition.
   */
  void set_neumann_boundary_condition(const types::boundary_id  boundary_id,
                                      const std::shared_ptr<TensorFunction<1, dim>> &function
                                        = std::shared_ptr<TensorFunction<1, dim>>(),
                                      const bool  time_dependent_bc = false);

  /*!
   * @brief Set a boundary condition on the tangential components of the field.
   */
  void set_normal_component_boundary_condition(const types::boundary_id  boundary_id,
                                               const std::shared_ptr<Function<dim>> &function
                                                 = std::shared_ptr<Function<dim>>(),
                                               const bool  time_dependent_bc = false);

  /*!
   * @brief Set a boundary condition on the tangential components of the field.
   */
  void set_tangential_component_boundary_condition(const types::boundary_id  boundary_id,
                                                   const std::shared_ptr<Function<dim>> &function
                                                     = std::shared_ptr<Function<dim>>(),
                                                   const bool  time_dependent_bc = false);

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
  virtual void apply_boundary_conditions(const bool check_regularity = true) override;

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
   * @brief This method evaluates the continuous finite element
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  Tensor<1, dim> point_value(const Point<dim>   &point,
                             const Mapping<dim> &external_mapping = MappingQ1<dim>()) const;

  /*!
   * @brief This method evaluates the gradient of the continuous finite element
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  Tensor<2, dim> point_gradient(const Point<dim>    &point,
                                const Mapping<dim>  &external_mapping = MappingQ1<dim>()) const;
};

template <int dim, typename VectorType>
const typename VectorBoundaryConditions<dim>::NeumannBCMapping &
FE_VectorField<dim, VectorType>::get_neumann_boundary_conditions() const
{
  return (static_cast<const VectorBoundaryConditions<dim> *>(this->boundary_conditions)
            ->neumann_bcs);
}


/*!
 * @struct FE_ScalarField
 *
 * @brief Numerical representation of a scalar field.
 */
template <int dim, typename VectorType = LinearAlgebra::MPI::Vector>
class FE_ScalarField: public FE_FieldBase<dim, VectorType>
{
public:
  /*!
   * @brief Constructor.
   */
  FE_ScalarField(const unsigned int         fe_degree,
                 const Triangulation<dim>  &triangulation,
                 const std::string         &name = "entity");

  /*!
   * @brief Copy constructor.
   */
  FE_ScalarField(const FE_ScalarField<dim, VectorType> &entity,
                 const std::string                     &new_name = "entity");

  const typename ScalarBoundaryConditions<dim>::NeumannBCMapping &
  get_neumann_boundary_conditions() const;

  /*!
   * @brief Set a Neumann boundary condition.
   */
  void set_neumann_boundary_condition(const types::boundary_id  boundary_id,
                                      const std::shared_ptr<Function<dim>> &function
                                       = std::shared_ptr<Function<dim>>(),
                                      const bool time_dependent_bc = false);

  /*!
   * @brief Set a datum boundary condition.
   */
  void set_datum_boundary_condition();

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
  virtual void apply_boundary_conditions(const bool check_regularity = true) override;

  /*!
   * @brief This method evaluates the continuous finite element
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  typename FE_FieldBase<dim, VectorType>::value_type
  point_value(const Point<dim>    &point,
              const Mapping<dim>  &external_mapping = MappingQ1<dim>()) const;

  /*!
   * @brief This method evaluates the gradient of the continuous finite element
   * field at the given point.
   *
   * @details It catches the value obtained by the processor who owns
   * the point while ignoring the rest. It also checks if the point
   * is inside the domain.
   */
  Tensor<1, dim> point_gradient(const Point<dim>    &point,
                                const Mapping<dim>  &external_mapping = MappingQ1<dim>()) const;

};

template <int dim, typename VectorType>
inline const typename ScalarBoundaryConditions<dim>::NeumannBCMapping &
FE_ScalarField<dim, VectorType>::get_neumann_boundary_conditions() const
{
  return (static_cast<const ScalarBoundaryConditions<dim> *>(this->boundary_conditions)
            ->neumann_bcs);
}


} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_FINITE_ELEMENT_FIELD_H_ */
