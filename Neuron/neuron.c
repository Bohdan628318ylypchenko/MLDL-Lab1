#include "neuron.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <stdlib.h>

static void _grad(unsigned long n, double * x,
			      double ya, double yt, double l,
				  double * grad);

static void _norm(unsigned long n, double * v);

/// <summary>
/// Calculates neuron output.
/// </summary>
/// <param name="n"> Neuron dimension: weight count / input count (include shift). </param>
/// <param name="w"> Weights array (include shift. </param>
/// <param name="x"> Input array (include shift). </param>
/// <param name="l"> Activation function smoothing coefficient. </param>
/// <returns> Neuron output. </returns>
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

/// <summary>
/// Adjusts neuron weights by gradient of Error from w values.
/// </summary>
/// <param name="n"> Neuron dimension: weight count / input count (include shift). </param>
/// <param name="w"> Weights array (include shift). </param>
/// <param name="x"> Input array (include shift). </param>
/// <param name="ya"> Neuron output. </param>
/// <param name="yt"> Expected output. </param>
/// <param name="l"> Activation function smoothing coefficient: o(s) = 1 / (1 + e ^ (-l * s)) </param>
/// <param name="a"> Weight adjustment length coefficient: wn = wc - a * ngrad </param>
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

/// <summary>
/// Calculates gradient of E(W).
/// Result is stored in grad.
/// </summary>
/// <param name="n"> Dimension. </param>
/// <param name="x"> Neuron input. </param>
/// <param name="ya"> Neuron output. </param>
/// <param name="yt"> Expected output. </param>
/// <param name="l"> Activation function smoothing coefficient. </param>
/// <param name="grad"> Array to store gradient coordinates in. </param>
static void _grad(unsigned long n, double * x,
				  double ya, double yt, double l,
				  double * grad)
{
	for (unsigned long i = 0; i < n; i++)
		grad[i] = (ya - yt)         // dE/dya
	            * l * ya * (1 - ya) // dya/ds
		        * x[i];             // ds/dwi;
}

/// <summary>
/// Normalizes given vector.
/// </summary>
/// <param name="n"> Vector dimension. </param>
/// <param name="v"> Vector values. </param>
static void _norm(unsigned long n, double * v)
{
	// vector module
	double m = 0;
	for (unsigned long i = 0; i < n; i++)
		m += v[i] * v[i];
	m = sqrt(m);

	// normalize
	for (unsigned long i = 0; i < n; i++)
		v[i] /= m;
}
