#include "neuron.h"

#include <math.h>
#include <stdio.h>

#define n 3
#define l 1
#define a 0.1
#define PRECISION 0.01
#define MAX_EPOCH_COUNT 100

int main()
{
	double w[n] = { 0.1, 0.5, 0.4 };
	double x[n] = { 0.2, 0.7, 1.0 };
	double yt = 0.9;

	printf("x0 = %lf, x1 = %lf, x_shift = %lf\n", x[0], x[1], x[2]);

	double ya;
	for (int i = 0; i < MAX_EPOCH_COUNT; i++)
	{
		printf("Iteration %d | w0 = %lf, w1 = %lf, w_shift = %lf | ", i, w[0], w[1], w[2]);

		ya = neuron_activate(n, w, x, l);
		printf("ya = %lf\n", ya);

		if (fabs(ya - yt) <= PRECISION)
			break;

		neuron_adjust_weights(n, w, x, ya, yt, l, a);
	}
}