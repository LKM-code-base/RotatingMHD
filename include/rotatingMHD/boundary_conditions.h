#ifndef INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_
#define INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_

#include <rotatingMHD/global.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/distributed/tria.h>

#include <algorithm>
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
   * @brief Constructor.
   */
  PeriodicBoundaryData(const types::boundary_id first_boundary,
                       const types::boundary_id second_boundary,
                       const unsigned int       direction,
                       const FullMatrix<double> rotation_matrix = FullMatrix<double>(IdentityMatrix(dim)),
                       const Tensor<1,dim>      offset = Tensor<1,dim>());

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
  tangential_flux,
  datum_at_boundary
};

/*!
 * @struct BoundaryConditionsBase
 * @brief A structure containing all instances related to the
 * boundary conditions which are independent
 * of the rank of the tensor field.
 */
template <int dim>
class BoundaryConditionsBase
{
public:
  /*!
   * @brief A typedef for a mapping using boundary ids as keys
   * and shared pointers to functions as values.
   */
  using BCMapping = typename std::map<types::boundary_id, std::shared_ptr< Function<dim> > >;

  /*!
   * @brief Constructor.
   */
  BoundaryConditionsBase(const Triangulation<dim> &triangulation,
                         const unsigned int n_components = 1);

  /*!
   * @brief A mapping of boundary ids on which Dirichlet boundary
   * conditions are set and their respective function.
   */
  typename BoundaryConditionsBase<dim>::BCMapping  dirichlet_bcs;

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

  /*!
   * Extract boundary ids from the triangulation.
   */
  void extract_boundary_ids();

  /*!
   * @brief Returns a vector containing the boundary indicators of the
   * unconstrained boundaries.
   */
  std::vector<types::boundary_id> get_unconstrained_boundary_ids() const;

  /*!
   * @brief Returns a vector containing the boundary indicators of the
   * constrained boundaries.
   */
  std::vector<types::boundary_id> get_constrained_boundary_ids() const;

  /*!
   * @brief This method sets a Dirichlet boundary condition by adding pair of a
   * boundary id and a function to @ref BoundaryConditionsBase::dirichlet_bcs.
   *
   * @details If no function is explicitly passed, it assumes that a
   * homogeneous boundary condition should be applied on the given boundary.
   * It calls the @ref check_boundary_id method before adding the entry and marks
   * the boundary as time dependent according to the boolean passed.
   */
  void set_dirichlet_bc(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>(),
                         const bool                           time_dependent_bc
                          = false);

  /*!
   * @brief This method sets a periodic boundary condition by adding a
   * @ref PeriodicBoundaryData object to the member variable
   * @ref BoundaryConditionsBase::periodic_bcs.
   *
   * @details It calls the @ref check_boundary_id method before
   * adding the entry.
   */
  void set_periodic_bc(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix =
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = Tensor<1,dim>());

  /*!
   * @brief This method sets the time of the functions by calling their
   * respective methods in a loop.
   */
  virtual void set_time(const double time);

  /*!
   * @brief A method returning the value of @ref flag_regularity_guaranteed.
   */
  bool                            regularity_guaranteed() const;

  /*!
   * @brief A method returning the value of @ref flag_boundary_conditions_closed.
   */
  bool                            closed() const;

  /*!
   * @todo Sets the boundary condition by setting @ref flag_boundary_conditions_closed
   * to true.
   */
  void                            close();

  /*!
   * @brief This method clears all boundary conditions.
   *
   * @details This is a pure virtual method. Its implementation is
   * overriden in the child structs
   */
  virtual void                    clear();

  /*!
   * @brief This method copies the content of another @ref ScalarBoundaryConditions
   * instance
   */
  virtual void copy(const BoundaryConditionsBase<dim> &other);

  /*!
   * @brief A method which prints a summary of the boundary conditions which are
   * currently specified to the stream object @p stream.
   */
  virtual void print_summary(std::ostream &stream, const std::string &name = "") const;

protected:
  /*!
   * @brief Reference to the underlying triangulation.
   * @todo Change it to a shared_ptr if copy constructure are allowed
   * through the change.
   */
  const Triangulation<dim>                       &triangulation;

  /*!
   * @brief A vector containing all boundary indicators assigned to
   * boundary faces of active cells of the @ref triangulation.
   */
  std::vector<types::boundary_id>                 boundary_ids;

  /*!
   * @brief A vector containing all the boundary indicators of the
   * constrainted boundaries.
   */
  std::vector<types::boundary_id>                 constrained_boundaries;

  /*!
   * @brief Number of components.
   */
  const unsigned int                              n_components;
public:
  /*!
   * @brief A flag indicating whether the boundary indicators are to be
   * extracted.
   */
  bool                                            flag_extract_boundary_ids;
protected:
  /*!
   * @brief A flag indicating that boundary conditions fulfill the
   * necessary conditions for a well-posed problem.
   */
  bool                                            flag_regularity_guaranteed;

  /*!
   * @brief A flag indicating wether the boundary conditions are closed
   * or not.
   */
  bool                                            flag_boundary_conditions_closed;

  /*!
   * @brief A scalar zero function used for homogeneous boundary
   * conditions.
   */
  const std::shared_ptr<Function<dim>>  zero_function_ptr;

  /*!
   * @brief Auxiliary method adding function to a collection of
   * boundary conditions.
   */
  void add_function_to_bc_mapping
  (typename BoundaryConditionsBase<dim>::BCMapping &bcs,
   const BCType                                     bc_type,
   const types::boundary_id                         boundary_id,
   const std::shared_ptr<Function<dim>>            &function,
   const unsigned int                               n_function_components,
   const bool                                       time_dependent_bc);

  /*!
   * @brief Checks if the boundary passed by through @p boundary_id was already
   * constrained.
   *
   * @details It throws an exception if the boundary was constrained.
   */
  virtual void check_boundary_id(const types::boundary_id boundary_id);

};



template <int dim>
inline void BoundaryConditionsBase<dim>::extract_boundary_ids()
{
  if (!flag_extract_boundary_ids)
    return;
  boundary_ids = triangulation.get_boundary_ids();
  flag_extract_boundary_ids = false;
}



template <int dim>
inline std::vector<types::boundary_id> BoundaryConditionsBase<dim>::
get_constrained_boundary_ids() const
{
  return (constrained_boundaries);
}



template <int dim>
inline bool BoundaryConditionsBase<dim>::regularity_guaranteed() const
{
  return (flag_regularity_guaranteed);
}



template <int dim>
inline bool BoundaryConditionsBase<dim>::closed() const
{
  return (flag_boundary_conditions_closed);
}



/*!
 * @struct ScalarBoundaryConditions
 *
 * @brief A structure containing all the instances related to the
 * boundary conditions of a scalar field.
 */
template <int dim>
class ScalarBoundaryConditions : public BoundaryConditionsBase<dim>
{
public:
  using NeumannBCMapping = typename BoundaryConditionsBase<dim>::BCMapping;

  /*!
   * @brief Constructor.
   */
  ScalarBoundaryConditions(const Triangulation<dim> &triangulation);

  /*!
   * @brief A mapping of boundary ids on which Neumann boundary
   * conditions are set and their respective function.
   */
  typename ScalarBoundaryConditions<dim>::NeumannBCMapping neumann_bcs;

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
  void set_neumann_bc(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                         = std::shared_ptr<Function<dim>>(),
                       const bool                           time_dependent_bc
                         = false);

  /*!
   * @brief Sets an admissible local degree of freedom at the boundary
   * to zero
   */
  void  set_datum_at_boundary();

  /*!
   * @brief A method returning the value of @ref flag_datum_at_boundary.
   */
  bool  datum_at_boundary() const;

  /*!
   * @brief This method sets the time of the functions by calling their
   * respective methods in a loop.
   */
  virtual void set_time(const double time) override;

  /*!
   * @brief This method clears all boundary conditions.
   */
  virtual void clear() override;

  /*!
   * @brief This method copies the content of another @ref ScalarBoundaryConditions
   * instance
   */
  virtual void copy(const BoundaryConditionsBase<dim> &other) override;

  /*!
   * @brief A method which prints a summary of the boundary conditions which are
   * currently specified to the stream object @p stream.
   */
  virtual void print_summary(std::ostream &stream, const std::string &name) const;

protected:
  /*!
   * @brief Checks if the boundary passed by through @p boundary_id was already
   * constrained.
   *
   * @details It throws an exception if the boundary was constrained.
   */
  virtual void check_boundary_id(const types::boundary_id boundary_id);

private:
  /*!
   * @brief A flag indicating that a single degree of freedom is constrained
   * at the boundary. This is required to obtain a regular system matrix
   * in case of a pure Neumann problem.
   */
  bool  flag_datum_at_boundary;

};



template <int dim>
inline bool ScalarBoundaryConditions<dim>::datum_at_boundary() const
{
  return (flag_datum_at_boundary);
}



/*!
 * @struct VectorBoundaryConditions
 * @brief A structure containing all the instances related to the
 * boundary conditions of a vector field.
 */
template <int dim>
class VectorBoundaryConditions : public BoundaryConditionsBase<dim>
{
public:
  /*!
   * @brief A typdef for the a mapping using boundary ids as keys
   * and tensor functions shared pointers as value.
   */
  using NeumannBCMapping = typename std::map<types::boundary_id,
                                             std::shared_ptr< TensorFunction<1,dim> > >;

  /*!
   * @brief Constructor.
   */
  VectorBoundaryConditions(const Triangulation<dim> &triangulation);

  /*!
   * @brief A mapping of boundary ids on which Neumann boundary
   * conditions are set and their respective function.
   */
  typename VectorBoundaryConditions<dim>::NeumannBCMapping  neumann_bcs;

  /*!
   * @brief A mapping of boundary ids on which normal flux boundary
   * conditions are set and their respective functions.
   */
  typename BoundaryConditionsBase<dim>::BCMapping normal_flux_bcs;

  /*!
   * @brief A mapping of boundary ids on which tangential flux boundary
   * conditions are set and their respective functions.
   */
  typename BoundaryConditionsBase<dim>::BCMapping tangential_flux_bcs;

  /*!
   * @brief A method which prints a summary of the boundary conditions which are
   * currently specified to the stream object @p stream.
   */
  virtual void print_summary(std::ostream &stream, const std::string &name) const;

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
  void set_neumann_bc(const types::boundary_id                       boundary_id,
                       const std::shared_ptr<TensorFunction<1, dim>>  &function
                          = std::shared_ptr<TensorFunction<1, dim>>(),
                       const bool                                     time_dependent_bc
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
  void set_normal_flux_bc(const types::boundary_id             boundary_id,
                           const std::shared_ptr<Function<dim>> &function
                              = std::shared_ptr<Function<dim>>(),
                           const bool                           time_dependent_bc
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
  void set_tangential_flux_bc(const types::boundary_id             boundary_id,
                               const std::shared_ptr<Function<dim>> &function
                                  = std::shared_ptr<Function<dim>>(),
                               const bool                           time_dependent_bc
                                  = false);

  /*!
   * @brief This method sets the time of the functions by calling their
   * respective methods in a loop.
   */
  virtual void set_time(const double time) override;

  /*!
   * @brief This method clears all boundary conditions.
   */
  virtual void clear() override;

  /*!
   * @brief This method copies the content of another @ref VectorBoundaryConditions
   * instance.
   */
  virtual void copy(const BoundaryConditionsBase<dim> &other);

protected:
  /*!
   * @brief Checks if the boundary passed by through @p boundary_id was already
   * constrained.
   *
   * @details It throws an exception if the boundary was constrained.
   */
  virtual void check_boundary_id(const types::boundary_id boundary_id) override;


private:
  /*!
   * @brief A scalar zero tensor function used for homogeneous boundary
   * conditions.
   */
  const std::shared_ptr<TensorFunction<1, dim>>  zero_tensor_function_ptr
    = std::make_shared<ZeroTensorFunction<1,dim>>();

};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_ */
