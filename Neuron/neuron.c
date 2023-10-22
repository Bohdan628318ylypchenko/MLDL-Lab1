#include "neuron.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdlib.h>

static void _grad(unsigned long n, double * x,
			      double ya, double yt, double l,
				  double * grad);

static void _norm(unsigned long n, double * v);

double neuron_activate(unsigned long n, double * w, double * x, double l)
{
	// Activation function argument
	double s = 0;
	for (unsigned long i = 0; i < n; i++)
		s += w[i] * x[i];

	// Activation function
	double ya = 1.0 / (1.0 + exp(-1.0 * l * s));

	// Returning
	return ya;
}

void neuron_adjust_weights(unsigned long n,
						   double * w, double * x, 
				           double ya, double yt, double l, double a)
{
	// Calculate grad
	double * grad = (double *)malloc(n * sizeof(double));
	_grad(n, x, ya, yt, l, grad);

	// Normalize
	_norm(n, grad);

	// Adjust weight
	for (unsigned long i = 0; i < n; i++)
		w[i] -= a * grad[i];

	// Free resources
	free(grad);
}

static void _grad(unsigned long n, double * x,
				  double ya, double yt, double l,
				  double * grad)
{
	for (unsigned long i = 0; i < n; i++)
		grad[i] = (ya - yt)         // dE/dya
	            * l * ya * (1 - ya) // dya/ds
		        * x[i];             // ds/dwi;
}

static void _norm(unsigned long n, double * v)
{
	// grad module
	double m = 0;
	for (unsigned long i = 0; i < n; i++)
		m += v[i] * v[i];
	m = sqrt(m);

	// normalize
	for (unsigned long i = 0; i < n; i++)
		v[i] /= m;
}
