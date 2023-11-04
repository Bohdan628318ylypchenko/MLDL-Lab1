#include "nn.h"

/// <summary>
/// Writes neural network to stream in binary format.
/// </summary>
/// <param name="l"> NN lambda parameter. </param>
/// <param name="a"> NN alpha parameter. </param>
/// <param name="w12"> Weights of layer1 -> layer2 as matrix. </param>
/// <param name="w23"> Weights of layer2 -> layer3 as vector. </param>
/// <param name="f"> Output stream. </param>
void nn_fwrite(double l, double a,
			   double w12[L1COUNT + 1][L2COUNT], double w23[L2COUNT + 1],
			   FILE * f)
{
	fwrite(&l, sizeof(double), 1, f);
	fwrite(&a, sizeof(double), 1, f);
	for (int i = 0; i < L1COUNT + 1; i++)
		fwrite(w12[i], sizeof(double), L2COUNT, f);
	fwrite(w23, sizeof(double), L2COUNT + 1, f);
}

/// <summary>
/// Reads neural network from stream.
/// </summary>
/// <param name="l"> Pointer to read NN lambda parameter in. </param>
/// <param name="a"> Pointer to read NN alpha parameter in. </param>
/// <param name="w12"> Matrix to read layer1->layer2 weights in. </param>
/// <param name="w23"> Vector to read layer2->layer3 weights in. </param>
/// <param name="f"> Output stream. </param>
void nn_fread(double * l, double * a,
			  double w12[L1COUNT + 1][L2COUNT], double w23[L2COUNT + 1],
			  FILE * f)
{
	fread(l, sizeof(double), 1, f);
	fread(a, sizeof(double), 1, f);
	for (int i = 0; i < L1COUNT + 1; i++)
		fread(w12[i], sizeof(double), L2COUNT, f);
	fread(w23, sizeof(double), L2COUNT + 1, f);
}

/// <summary>
/// Writes neural network to stream in text format.
/// </summary>
/// <param name="l"> NN lambda parameter. </param>
/// <param name="a"> NN alpha parameter. </param>
/// <param name="w12"> Weights of layer1 -> layer2 as matrix. </param>
/// <param name="w23"> Weights of layer2 -> layer3 as vector. </param>
/// <param name="f"> Output stream. </param>
void nn_fprint(double l, double a,
			   double w12[L1COUNT + 1][L2COUNT], double w23[L2COUNT + 1],
			   FILE * stream)
{
	fprintf(stream, "l = %.4lf; a = %.4lf;\n", l, a);

	fprintf(stream, "w12:");
	for (int i = 0; i < L1COUNT + 1; i++)
	{
		fputs("\n|", stream);
		for (int j = 0; j < L2COUNT; j++)
		{
			fprintf(stream, " %.4lf |", w12[i][j]);
		}
	}

	fprintf(stream, "\nw23:\n|");
	for (int i = 0; i < L2COUNT + 1; i++)
	{
		fprintf(stream, " %.4lf |", w23[i]);
	}
	fputc('\n', stream);
}
