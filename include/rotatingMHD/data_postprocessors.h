#ifndef INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_
#define INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_

#include <deal.II/numerics/data_postprocessor.h>

namespace RMHD {

using namespace dealii;

/*!
 * @class PostprocessorScalarField
 *
 * @brief A postprocessor for a scalar finite element field.
 *
 * @details This postprocessor outputs the field itself and the gradient.
 */
template<int dim>
class PostprocessorScalarField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  PostprocessorScalarField(const std::string &name);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details Value and gradient of the field is written as an output.
   */
  virtual void evaluate_scalar_field(
    const DataPostprocessorInputs::Scalar<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;
};


/*!
 * @class SphericalPostprocessorScalarField
 *
 * @brief A postprocessor for a scalar finite element field.
 *
 * @details This postprocessor outputs the spherical components of the gradient
 * of the field .
 */
template<int dim>
class SphericalPostprocessorScalarField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  SphericalPostprocessorScalarField(const std::string &name);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details Only the spherical components of the gradient of the field is
   * written as an output.
   */
  virtual void evaluate_scalar_field(
    const DataPostprocessorInputs::Scalar<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;

  /*!
   * @brief Numerical tolerance used to check whether spherical components are
   * computed correctly.
   */
  const double        TOL{1e-12};
};


template<int dim>
class PostprocessorVectorField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  PostprocessorVectorField(const std::string &name);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details The field, its curl, the helicity and the invariants of the gradient
   * are written as an output.
   */
  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;
};


template<int dim>
class SphericalPostprocessorVectorField : public DataPostprocessor<dim>
{
public:
  /*!
   * @brief Default constructor specifying the name of the field.
   */
  SphericalPostprocessorVectorField(const std::string &name);

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::evaluate_scalar_field`.
   *
   * @details The spherical components of the field and its curl are written as
   * an output.
   */
  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_data_component_interpretation`.
   */
  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_names`.
   */
  virtual std::vector<std::string> get_names() const override;

  /*!
   * @brief Overloading `dealii::DataPostprocessorScalar<dim>::get_needed_update_flags`.
   */
  virtual UpdateFlags get_needed_update_flags() const override;

private:
  /*!
   * @brief Name of the scalar finite element field.
   */
  const std::string   name;

  /*!
   * @brief Numerical tolerance used to check whether spherical components are
   * computed correctly.
   */
  const double        TOL{1e-12};
};

}  // namespace RMHD


#endif /* INCLUDE_ROTATINGMHD_DATA_POSTPROCESSORS_H_ */
