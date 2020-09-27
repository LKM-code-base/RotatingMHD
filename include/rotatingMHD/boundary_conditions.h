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

template <int dim>
struct PeriodicBoundaryData
{
  std::pair<types::boundary_id,types::boundary_id>  boundary_pair;

  unsigned int                                      direction;

  FullMatrix<double>                                rotation_matrix;

  Tensor<1,dim>                                     offset;

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

template <int dim>
struct BoundaryConditionsBase
{
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;

  FunctionMapping                         dirichlet_bcs;
  
  FunctionMapping                         neumann_bcs;

  std::vector<PeriodicBoundaryData<dim>>  periodic_bcs;

  std::vector<types::boundary_id>         boundary_ids;
};

template <int dim>
struct ScalarBoundaryConditions : BoundaryConditionsBase<dim>
{

  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix = 
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = 
                                Tensor<1,dim>());

  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>());

  void set_neumann_bcs(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>());

  void set_time(const double &time);

  void clear();

private:

  void check_boundary_id(types::boundary_id boundary_id) const;

};

template <int dim>
struct VectorBoundaryConditions : BoundaryConditionsBase<dim>
{
  using FunctionMapping = std::map<types::boundary_id,
                                   std::shared_ptr<Function<dim>>>;

  FunctionMapping   normal_flux_bcs;

  FunctionMapping   tangential_flux_bcs;

  void set_periodic_bcs(const types::boundary_id  first_boundary,
                        const types::boundary_id  second_boundary,
                        const unsigned int        direction,
                        const FullMatrix<double>  rotation_matrix = 
                                FullMatrix<double>(IdentityMatrix(dim)),
                        const Tensor<1,dim>       offset = 
                                Tensor<1,dim>());

  void set_dirichlet_bcs(const types::boundary_id             boundary_id,
                         const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>());

  void set_neumann_bcs(const types::boundary_id             boundary_id,
                       const std::shared_ptr<Function<dim>> &function
                          = std::shared_ptr<Function<dim>>());

  void set_normal_flux_bcs(const types::boundary_id             boundary_id,
                           const std::shared_ptr<Function<dim>> &function
                              = std::shared_ptr<Function<dim>>());

  void set_tangential_flux_bcs(const types::boundary_id             boundary_id,
                               const std::shared_ptr<Function<dim>> &function
                                  = std::shared_ptr<Function<dim>>());

  void set_time(const double &time);

  void clear();

private:

  void check_boundary_id(types::boundary_id boundary_id) const;

};

} // namespace Entities

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_BOUNDARY_CONDITIONS_H_ */
