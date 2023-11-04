#pragma once

#include <stdio.h>

#define L1COUNT 2
#define L2COUNT 3

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
				 double o1[L1COUNT + 1], double o2[L2COUNT + 1], double * oa);

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
			   double o1[L1COUNT + 1], double o2[L2COUNT + 1]);

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
			   FILE * f);

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
			  FILE * f);

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
			   FILE * stream);
