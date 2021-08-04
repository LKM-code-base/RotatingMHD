#ifndef INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_
#define INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_

#include <deal.II/numerics/data_postprocessor.h>

namespace RMHD {

using namespace dealii;

template<int dim>
class ScalarFieldPostprocessor : public DataPostprocessor<dim>
{
  ScalarFieldPostprocessor(
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

  bool                postprocess_spherical;
};


template<int dim>
class VectorFieldPostprocessor : public DataPostprocessor<dim>
{
  VectorFieldPostprocessor(
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

  bool                postprocess_spherical;
};

}  // namespace RMHD


#endif /* INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_ */
