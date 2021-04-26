/*
 * exceptions.h
 *
 *  Created on: May 16, 2019
 *      Author: sg
 */

#ifndef INCLUDE_EXCEPTIONS_H_
#define INCLUDE_EXCEPTIONS_H_

#include <deal.II/base/exceptions.h>

namespace GeometryExceptions
{

using namespace dealii;

DeclException1(ExcNegativeRadius,
               double,
               << "The radius, r = " << arg1
               << ", is not positive.");

DeclException1(ExcPolarAngleRange,
               double,
               << "The polar angle theta = " << arg1
               << ", is not in the half-open range [0,pi).");

DeclException1(ExcAzimuthalAngleRange,
               double,
               << "The azimuthal angle, phi =  " << arg1
               << ", is not in the half-open range [0,2*pi).");

}  // namespace GeometryExceptions


#endif /* INCLUDE_EXCEPTIONS_H_ */