#pragma once

double neuron_activate(unsigned long n, double * w, double * x, double l);

void neuron_adjust_weights(unsigned long n,
						   double * w, double * x,
						   double ya, double yt, double l, double a);
