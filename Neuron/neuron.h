#pragma once

/// <summary>
/// Calculates neuron output.
/// </summary>
/// <param name="n"> Neuron dimension: weight count / input count (include shift). </param>
/// <param name="w"> Weights array (include shift. </param>
/// <param name="x"> Input array (include shift). </param>
/// <param name="l"> Activation function smoothing coefficient. </param>
/// <returns> Neuron output. </returns>
double neuron_activate(unsigned long n, double * w, double * x, double l);

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
						   double ya, double yt, double l, double a);
