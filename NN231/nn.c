#include "nn.h"

#define _USE_MATH_DEFINES
#include <math.h>

#define SIGMA(x, l) 1.0 / (1.0 + exp(-1.0 * l * x))

/// <summary>
/// NN activation implementation.
/// </summary>
/// <param name="x"> Input signal vector. </param>
/// <param name="w12"> layer1->layer2 weights as matrix. </param>
/// <param name="w23"> layer2->layer3 weights as vector. </param>
/// <param name="l"> NN lambda parameter. </param>
/// <param name="o1"> Vector to write layer1 output. </param>
/// <param name="o2"> Vector to write layer2 output. </param>
/// <param name="oa"> Pointer to write layer3 output. </param>
void nn_activate(double x[L1COUNT],
			     double w12[L1COUNT + 1][L2COUNT], double w23[L2COUNT + 1], double l,
			     double o1[L1COUNT + 1], double o2[L2COUNT + 1], double * oa)
{
	// 1st layer output + shift
	for (int i = 0; i < L1COUNT; i++)
		o1[i] = x[i];
	o1[L1COUNT] = 1.0;

	// Sum for 2nd layer
	double s2[L2COUNT] = { 0.0, 0.0, 0.0 };
	for (int i = 0; i < L2COUNT; i++)
		for (int j = 0; j < L1COUNT + 1; j++)
			s2[i] += o1[j] * w12[j][i];

	// 2nd layer output + shift
	for (int i = 0; i < L2COUNT; i++)
		o2[i] = SIGMA(s2[i], l);
	o2[L2COUNT] = 1.0;

	// Sum for 3rd layer
	double s3 = 0;
	for (int i = 0; i < L2COUNT + 1; i++)
		s3 += o2[i] * w23[i];

	// Final output
	*oa = SIGMA(s3, l);
}

/// <summary>
/// Does 1 nn weights adjustment based on layer1, layer2, layer3 output
/// and expected NN output.
/// </summary>
/// <param name="oa"> Layer3 output. </param>
/// <param name="ot"> Expected NN output. </param>
/// <param name="w12"> layer1->layer2 weights as matrix. </param>
/// <param name="w23"> layer2->layer3 weights as vector. </param>
/// <param name="l"> NN lambda parameter. </param>
/// <param name="a"> NN alpha parameter. </param>
/// <param name="o1"> Layer1 output. </param>
/// <param name="o2"> Layer2 output. </param>
void nn_adjust(double oa, double ot,
			   double w12[L1COUNT + 1][L2COUNT], double w23[L2COUNT + 1], double l, double a,
			   double o1[L1COUNT + 1], double o2[L2COUNT + 1])
{
	// Common part
	double cdelta = l * oa * (1.0 - oa) * (oa - ot);
	
	// 2l -> 3l deltas
	double deltas23[L2COUNT + 1];
	for (int i = 0; i < L2COUNT + 1; i++)
		deltas23[i] = o2[i] * cdelta;

	// 1l -> 2l deltas
	double deltas12[L1COUNT + 1][L2COUNT];
	for (int i = 0; i < L1COUNT + 1; i++)
		for (int j = 0; j < L2COUNT; j++)
			deltas12[i][j] = o1[i] * l * o2[j] * (1 - o2[j]) * w23[j] * cdelta;

	// 2l -> 3l adjustment
	for (int i = 0; i < L2COUNT + 1; i++)
		w23[i] -= (a * deltas23[i]);

	// 1l -> 2l adjustment
	for (int i = 0; i < L1COUNT + 1; i++)
		for (int j = 0; j < L2COUNT; j++)
			w12[i][j] -= (a * deltas12[i][j]);
}
