#include <rotatingMHD/basic_parameters.h>

#include <deal.II/base/conditional_ostream.h>

#include <iomanip>
#include <iostream>

namespace RMHD
{

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

namespace internal
{

constexpr char header[] = "+------------------------------------------+"
                      "----------------------+";

constexpr size_t column_width[2] ={ 40, 20 };

constexpr size_t line_width = 63;

template<typename Stream, typename A>
void add_line(Stream  &stream, const A line)
{
  stream << "| "
         << std::setw(line_width)
         << line
         << " |"
         << std::endl;
}

template<typename Stream, typename A, typename B>
void add_line(Stream  &stream, const A first_column, const B second_column)
{
  stream << "| "
         << std::setw(column_width[0]) << first_column
         << " | "
         << std::setw(column_width[1]) << second_column
         << " |"
         << std::endl;
}

template<typename Stream>
void add_header(Stream  &stream)
{
  stream << std::left << header << std::endl;
}

} // internal

} // namespace RunTimeParameters

} // namespace RMHD

template void RMHD::RunTimeParameters::internal::add_header
(std::ostream &);
template void RMHD::RunTimeParameters::internal::add_header
(dealii::ConditionalOStream &);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[]);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, std::string);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[]);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const double);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const double);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const double);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const double);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const char[]);

