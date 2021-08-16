#ifndef INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_
#define INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_

#include <deal.II/numerics/data_postprocessor.h>

namespace RMHD {

using namespace dealii;

template<int dim>
class PostprocessorScalarField : public DataPostprocessor<dim>
{
public:
  PostprocessorScalarField(
    const std::string &name,
    const unsigned int partition = numbers::invalid_unsigned_int);

  virtual void evaluate_scalar_field(
    const DataPostprocessorInputs::Scalar<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<std::string> get_names() const override;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual UpdateFlags get_needed_update_flags() const override;

private:
  const std::string   name;

  const unsigned int  partition;
};


template<int dim>
class SphericalPostprocessorScalarField : public DataPostprocessor<dim>
{
public:
  SphericalPostprocessorScalarField(
    const std::string &name,
    const unsigned int partition = numbers::invalid_unsigned_int);

  virtual void evaluate_scalar_field(
    const DataPostprocessorInputs::Scalar<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<std::string> get_names() const override;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual UpdateFlags get_needed_update_flags() const override;

private:
  const std::string   name;

  const unsigned int  partition;

  const double        TOL{1e-12};
};


template<int dim>
class PostprocessorVectorField : public DataPostprocessor<dim>
{
public:
  PostprocessorVectorField(
    const std::string &name,
    const unsigned int partition = numbers::invalid_unsigned_int);

  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual std::vector<std::string> get_names() const override;

  virtual UpdateFlags get_needed_update_flags() const override;

private:
  const std::string   name;

  const unsigned int  partition;
};


template<int dim>
class SphericalPostprocessorVectorField : public DataPostprocessor<dim>
{
public:
  SphericalPostprocessorVectorField(
    const std::string &name,
    const unsigned int partition = numbers::invalid_unsigned_int);

  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual std::vector<std::string> get_names() const override;

  virtual UpdateFlags get_needed_update_flags() const override;

private:
  const std::string   name;

  const unsigned int  partition;

  const double        TOL{1e-12};
};

}  // namespace RMHD


#endif /* INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_ */
