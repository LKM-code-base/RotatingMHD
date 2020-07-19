/*
 * data_storage.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace Step35
{

using namespace dealii;

namespace RunTimeParameters
{

enum class Method
{
	standard,
	rotational
};

class Data_Storage
{
public:
	Data_Storage();

	void read_data(const std::string &filename);

	Method form;

	double dt;
	double initial_time;
	double final_time;

	double Reynolds;

	unsigned int n_global_refines;

	unsigned int pressure_degree;

	unsigned int vel_max_iterations;
	unsigned int vel_Krylov_size;
	unsigned int vel_off_diagonals;
	unsigned int vel_update_prec;
	double       vel_eps;
	double       vel_diag_strength;

	bool         verbose;
	unsigned int output_interval;

protected:
	ParameterHandler prm;
};

} // namespace RunTimeParameters

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
