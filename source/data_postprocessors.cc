#include <rotatingMHD/data_postprocessors.h>
#include <rotatingMHD/exceptions.h>

#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>


namespace RMHD
{

template <int dim>
PostprocessorScalarField<dim>::PostprocessorScalarField
(const std::string &name)
:
DataPostprocessor<dim>(),
name(name)
{}


template <int dim>
void PostprocessorScalarField<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{

  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // solution value
    computed_quantities[q](cnt) = inputs.solution_values[q];
    cnt += 1;
    // gradient of the solution
    for (unsigned int d=0; d<dim; ++d)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][d];
      cnt += 1;
    }
  }
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostprocessorScalarField<dim>::get_data_component_interpretation() const
{
  // solution
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(1, DataComponentInterpretation::component_is_scalar);
  // gradient of the solution
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

  return (component_interpretation);
}


template <int dim>
std::vector<std::string> PostprocessorScalarField<dim>::get_names() const
{
  // the solution
  std::vector<std::string> solution_names(1, name);
  // gradient of the solution
  for (unsigned int d=0; d<dim; ++d)
    solution_names.emplace_back(name + "_gradient");

  return (solution_names);
}


template <int dim>
UpdateFlags PostprocessorScalarField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients;
}


template <int dim>
SphericalPostprocessorScalarField<dim>::SphericalPostprocessorScalarField
(const std::string &name)
:
DataPostprocessor<dim>()
{}


template <int dim>
void SphericalPostprocessorScalarField<dim>::evaluate_scalar_field
(const DataPostprocessorInputs::Scalar<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{

  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    for (unsigned int i=0; i<computed_quantities[q].size(); ++i)
      computed_quantities[q](i) = 0;

    unsigned int cnt = 0;

    // spherical coordinates
    const std::array<double, dim> scoord
      = GeometricUtilities::Coordinates::to_spherical(inputs.evaluation_points[q]);

    // spherical basis vectors
    std::vector<Tensor<1,dim>> spherical_basis_vectors(dim);
    if constexpr(dim == 2)
    {
      const double phi{scoord[1]};
      // radial basis vector
      spherical_basis_vectors[0][0] = cos(phi);
      spherical_basis_vectors[0][1] = sin(phi);
      // azimuthal basis vector
      spherical_basis_vectors[1][0] = -sin(phi);
      spherical_basis_vectors[1][1] = cos(phi);
    }
    else if constexpr(dim == 3)
    {
      const double theta{scoord[2]};
      const double phi{scoord[2]};
      // radial basis vector
      spherical_basis_vectors[0][0] = cos(phi) * sin(theta);
      spherical_basis_vectors[0][1] = sin(phi) * sin(theta);
      spherical_basis_vectors[0][2] = cos(theta);
      // polar basis vector
      spherical_basis_vectors[1][0] = cos(phi) * cos(theta);
      spherical_basis_vectors[1][1] = sin(phi) * cos(theta);
      spherical_basis_vectors[1][2] = -sin(theta);
      // azimuthal basis vector
      spherical_basis_vectors[2][0] = -sin(phi);
      spherical_basis_vectors[2][1] = cos(phi);
      spherical_basis_vectors[2][2] = 0.0;
    }

    // spherical components of the solution
    double gradient_magnitude = 0;
    for (unsigned int d=0; d<dim; ++d)
    {
      for (unsigned int c=0; c<dim; ++c)
        computed_quantities[q](cnt+c) += spherical_basis_vectors[c][d] * inputs.solution_gradients[q][d];
      gradient_magnitude += inputs.solution_gradients[q][d] * inputs.solution_gradients[q][d];
    }
    gradient_magnitude = sqrt(gradient_magnitude);
    for (unsigned int c=0; c<dim; ++c)
      Assert(computed_quantities[q](cnt+c) <= gradient_magnitude + TOL,
             ExcLowerRangeType<double>(computed_quantities[q](cnt+c),
                                       gradient_magnitude));
    cnt += dim;
  }
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
SphericalPostprocessorScalarField<dim>::get_data_component_interpretation() const
{
  // spherical components of the gradient
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim, DataComponentInterpretation::component_is_scalar);

  return (component_interpretation);
}


template <int dim>
std::vector<std::string> SphericalPostprocessorScalarField<dim>::get_names() const
{
  std::vector<std::string> solution_names;

  // spherical components of the gradient
  if constexpr(dim == 2)
  {
    solution_names.emplace_back("radial_" + name + "_gradient");
    solution_names.emplace_back("azimuthal_" + name + "_gradient");
  }
  else if constexpr(dim == 3)
  {
    solution_names.emplace_back("radial_" +  name + "_gradient");
    solution_names.emplace_back("polar_" + name + "_gradient");
    solution_names.emplace_back("azimuthal_" + name + "_gradient");
  }

  return (solution_names);
}


template <int dim>
UpdateFlags SphericalPostprocessorScalarField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients|update_quadrature_points;
}


template <int dim>
PostprocessorVectorField<dim>::PostprocessorVectorField
(const std::string &name)
:
DataPostprocessor<dim>(),
name(name)
{}


template <int dim>
void PostprocessorVectorField<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{

  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    unsigned int cnt = 0;

    // solution values
    for (unsigned int d=0; d<dim; ++d)
    {
      computed_quantities[q](cnt) = inputs.solution_values[q](d);
      cnt += 1;
    }
    // curl of the solution
    if constexpr(dim == 2)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][1][0]
                                  - inputs.solution_gradients[q][0][1];
      cnt += 1;
    }
    else if constexpr(dim == 3)
    {
      computed_quantities[q](cnt) = inputs.solution_gradients[q][2][1]
                                  - inputs.solution_gradients[q][1][2];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][0][2]
                                  - inputs.solution_gradients[q][2][0];
      computed_quantities[q](cnt) = inputs.solution_gradients[q][1][0]
                                  - inputs.solution_gradients[q][0][1];
      cnt += 3;
    }
    // helicity of the solution
    if constexpr(dim == 3)
    {
      Tensor<1,dim> solution_curl;
      solution_curl[0] = inputs.solution_gradients[q][2][1]
                       - inputs.solution_gradients[q][1][2];
      solution_curl[1] = inputs.solution_gradients[q][0][2]
                       - inputs.solution_gradients[q][2][0];
      solution_curl[2] = inputs.solution_gradients[q][1][0]
                       - inputs.solution_gradients[q][0][1];

      computed_quantities[q](cnt) = 0;
      for (unsigned int d=0; d<dim; ++d)
        computed_quantities[q](cnt) += solution_curl[d] * inputs.solution_values[q](d);
      cnt += 1;
    }

    Tensor<2,dim> solution_gradient;
    for (unsigned int c=0; c<dim; ++c)
      for (unsigned int d=0; d<=c; ++d)
        solution_gradient[TableIndices<2>(c,d)] = inputs.solution_gradients[q][c][d];
    SymmetricTensor<2,dim> symmetric_solution_gradient = symmetrize(solution_gradient);
    // first principle invariant
    computed_quantities[q](cnt) = first_invariant(symmetric_solution_gradient);
    cnt += 1;
    // second principle invariant
    computed_quantities[q](cnt) = second_invariant(symmetric_solution_gradient);
    cnt += 1;
    // third principle invariant
    computed_quantities[q](cnt) = third_invariant(symmetric_solution_gradient);
    cnt += 1;
  }
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostprocessorVectorField<dim>::get_data_component_interpretation() const
{
  // solution
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  // curl of the solution
  if constexpr(dim == 2)
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  // helicity of the solution
  if constexpr(dim == 3)
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // first principle invariant
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // second principle invariant
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // third principle invariant
  component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  return (component_interpretation);
}


template <int dim>
std::vector<std::string> PostprocessorVectorField<dim>::get_names() const
{
  // the solution
  std::vector<std::string> solution_names(dim, name);
  // curl of the solution
  if constexpr(dim == 2)
    solution_names.emplace_back(name + "_curl");
  else if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      solution_names.emplace_back(name + "_curl");
  // helicity of the solution
  if constexpr(dim == 3)
    solution_names.emplace_back(name + "_helicity");
  // first principle invariant
  solution_names.emplace_back(name + "_gradient_1st_invariant");
  // second principle invariant
  solution_names.emplace_back(name + "_gradient_2nd_invariant");
  // third principle invariant
  solution_names.emplace_back(name + "_gradient_3rd_invariant");

  return (solution_names);
}


template <int dim>
UpdateFlags PostprocessorVectorField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients;
}


template <int dim>
SphericalPostprocessorVectorField<dim>::SphericalPostprocessorVectorField
(const std::string &name)
:
DataPostprocessor<dim>(),
name(name)
{}


template <int dim>
void SphericalPostprocessorVectorField<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
 std::vector<Vector<double>> &computed_quantities) const
{
  const unsigned int n_quadrature_points = inputs.solution_values.size();
  Assert(computed_quantities.size() == n_quadrature_points,
         ExcDimensionMismatch(computed_quantities.size(),
                              n_quadrature_points));

  for (unsigned int q=0; q<n_quadrature_points; ++q)
  {
    for (unsigned int i=0; i<computed_quantities[q].size(); ++i)
      computed_quantities[q](i) = 0;
    unsigned int cnt = 0;

    // spherical coordinates
    const std::array<double, dim> scoord
      = GeometricUtilities::Coordinates::to_spherical(inputs.evaluation_points[q]);

    // spherical basis vectors
    std::vector<Tensor<1,dim>> spherical_basis_vectors(dim);
    if constexpr(dim == 2)
    {
      const double phi{scoord[1]};
      // radial basis vector
      spherical_basis_vectors[0][0] = cos(phi);
      spherical_basis_vectors[0][1] = sin(phi);
      // azimuthal basis vector
      spherical_basis_vectors[1][0] = -sin(phi);
      spherical_basis_vectors[1][1] = cos(phi);
    }
    else if constexpr(dim == 3)
    {
      const double theta{scoord[2]};
      const double phi{scoord[2]};
      // radial basis vector
      spherical_basis_vectors[0][0] = cos(phi) * sin(theta);
      spherical_basis_vectors[0][1] = sin(phi) * sin(theta);
      spherical_basis_vectors[0][2] = cos(theta);
      // polar basis vector
      spherical_basis_vectors[1][0] = cos(phi) * cos(theta);
      spherical_basis_vectors[1][1] = sin(phi) * cos(theta);
      spherical_basis_vectors[1][2] = sin(theta);
      // azimuthal basis vector
      spherical_basis_vectors[2][0] = -sin(phi);
      spherical_basis_vectors[2][1] = cos(phi);
      spherical_basis_vectors[2][2] = 0.0;
    }

    // spherical components of the solution
    double solution_magnitude = 0;
    for (unsigned int d=0; d<dim; ++d)
    {
      for (unsigned int c=0; c<dim; ++c)
        computed_quantities[q](cnt+c) += spherical_basis_vectors[c][d] * inputs.solution_values[q](d);
      solution_magnitude += inputs.solution_values[q](d) * inputs.solution_values[q](d);
    }
    solution_magnitude = sqrt(solution_magnitude);
    for (unsigned int c=0; c<dim; ++c)
      Assert(computed_quantities[q](cnt+c) <= solution_magnitude + TOL,
             ExcLowerRangeType<double>(computed_quantities[q](cnt+c),
                                       solution_magnitude));
    cnt += dim;

    // spherical components of the curl
    if constexpr(dim == 3)
    {
      Tensor<1,dim> solution_curl;
      solution_curl[0] = inputs.solution_gradients[q][2][1]
                       - inputs.solution_gradients[q][1][2];
      solution_curl[1] = inputs.solution_gradients[q][0][2]
                       - inputs.solution_gradients[q][2][0];
      solution_curl[2] = inputs.solution_gradients[q][1][0]
                       - inputs.solution_gradients[q][0][1];

      for (unsigned int c=0; c<dim; ++c)
        for (unsigned int d=0; d<dim; ++d)
          computed_quantities[q](cnt+c) += spherical_basis_vectors[c][d] * solution_curl[d];

      const double curl_magnitude{solution_curl.norm()};
      for (unsigned int c=0; c<dim; ++c)
        Assert(computed_quantities[q](cnt+c) <= curl_magnitude + TOL,
               ExcLowerRangeType<double>(computed_quantities[q](cnt+c),
                                         curl_magnitude));
      cnt += 3;
    }
  }
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
SphericalPostprocessorVectorField<dim>::get_data_component_interpretation() const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation;

  // spherical components of the solution
  for (unsigned int d=0; d<dim; ++d)
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  // spherical components of the curl
  if constexpr(dim == 3)
    for (unsigned int d=0; d<dim; ++d)
      component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  return (component_interpretation);
}


template <int dim>
std::vector<std::string> SphericalPostprocessorVectorField<dim>::get_names() const
{
  std::vector<std::string> solution_names;

  // spherical components of the solution
  if constexpr(dim == 2)
  {
    solution_names.emplace_back("radial_" + name);
    solution_names.emplace_back("azimuthal_" + name);
  }
  else if constexpr(dim == 3)
  {
    solution_names.emplace_back("radial_" +  name);
    solution_names.emplace_back("polar_" + name);
    solution_names.emplace_back("azimuthal_" + name);
  }

  // spherical components of the curl
  if constexpr(dim == 3)
  {
    solution_names.emplace_back("radial_" + name + "_curl");
    solution_names.emplace_back("polar_" + name + "_curl");
    solution_names.emplace_back("azimuthal_" + name + "_curl");
  }
  return (solution_names);
}


template <int dim>
UpdateFlags SphericalPostprocessorVectorField<dim>::get_needed_update_flags() const
{
  return update_values|update_gradients|update_quadrature_points;
}


}  // namespace RMHD

// explicit instantiations
template class RMHD::PostprocessorScalarField<2>;
template class RMHD::PostprocessorScalarField<3>;

template class RMHD::SphericalPostprocessorScalarField<2>;
template class RMHD::SphericalPostprocessorScalarField<3>;

template class RMHD::PostprocessorVectorField<2>;
template class RMHD::PostprocessorVectorField<3>;

template class RMHD::SphericalPostprocessorVectorField<2>;
template class RMHD::SphericalPostprocessorVectorField<3>;
