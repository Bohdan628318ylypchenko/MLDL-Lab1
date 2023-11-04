#include "nn.h"

#include <stdlib.h>

#define EOI '\n'
#define INIT_LEN 16
#define INCR_COEF 2
#define MIN_LEFTOVER 1

#define USAGE_MSG "Usage:\n [n]ew;\n [t]rain a l example-path epoch-count;\n [v]alidate example-path;\n [s]ave path;\n [l]oad path;\n [p]rint;\n [r]un x1 x2;\n [u]sage;\n [e]xit;"
#define COMMAND_PROMT "Command: "
#define ERR_FOPEN "Can't open file %s.\n"

static char * finput_al(FILE * stream);

int main(void)
{
	double l = 1, a = 0.1;
	double w12[L1COUNT + 1][L2COUNT];
	double w23[L2COUNT + 1];

	double x[L1COUNT];
	double o1[L1COUNT + 1];
	double o2[L2COUNT + 1];
	double oa, ot;

	double ** examples;

	char examples_path[51];
	int epoch_count = 0;
	int example_count = 0;

	char * input = NULL, * args = NULL;
	FILE * f = NULL;

	puts(USAGE_MSG);
	while (1)
	{
		printf(COMMAND_PROMT);
		input = finput_al(stdin);

		switch (input[0])
		{
			case 'n':
				for (int i = 0; i < L1COUNT + 1; i++)
					for (int j = 0; j < L2COUNT; j++)
						w12[i][j] = 0.1;
				for (int i = 0; i < L2COUNT + 1; i++)
					w23[i] = 0.1;
				break;
			case 't':
				args = input + 2;
				sscanf_s(args, "%lf %lf %50s %d", &a, &l, examples_path, sizeof(examples_path), &epoch_count);
				fopen_s(&f, examples_path, "r");
				if (f == NULL)
				{
					fprintf(stderr, ERR_FOPEN, examples_path);
					break;
				}
				fscanf_s(f, "%d", &example_count);
				examples = (double **)malloc(example_count * sizeof(double *));
				for (int i = 0; i < example_count; i++)
				{
					examples[i] = (double *)malloc(3 * sizeof(double));
					fscanf_s(f, "%lf %lf %lf", &(examples[i][0]), &(examples[i][1]), &(examples[i][2]));
				}
				for (int i = 0; i < epoch_count; i++)
				{
					for (int j = 0; j < example_count; j++)
					{
						x[0] = examples[j][0];
						x[1] = examples[j][1];
						ot = examples[j][2];

						nn_activate(x, w12, w23, l, o1, o2, &oa);
						
						nn_adjust(oa, ot, w12, w23, l, a, o1, o2);
					}
				}
				for (int i = 0; i < example_count; i++)
				{
					free(examples[i]);
				}
				free(examples);
				fclose(f);
				break;
			case 'v':
				args = input + 2;
				sscanf_s(args, "%50s", examples_path, sizeof(examples_path));
				fopen_s(&f, examples_path, "r");
				if (f == NULL)
				{
					fprintf(stderr, ERR_FOPEN, examples_path);
					break;
				}
				fscanf_s(f, "%d", &example_count);
				examples = (double **)malloc(example_count * sizeof(double *));
				for (int i = 0; i < example_count; i++)
				{
					examples[i] = (double *)malloc(3 * sizeof(double));
					fscanf_s(f, "%lf %lf %lf", &(examples[i][0]), &(examples[i][1]), &(examples[i][2]));
				}
				double total_error = 0;
				for (int j = 0; j < example_count; j++)
				{
					x[0] = examples[j][0];
					x[1] = examples[j][1];
					ot = examples[j][2];

					nn_activate(x, w12, w23, l, o1, o2, &oa);

					printf("oa = %lf; ot = %lf\n", oa, ot);
					total_error += (oa - ot) * (oa - ot);
				}
				printf("error = %lf\n", total_error);
				printf("average_error = %lf\n", total_error / (double)example_count);
				for (int i = 0; i < example_count; i++)
				{
					free(examples[i]);
				}
				free(examples);
				fclose(f);
				break;
			case 's':
				args = input + 2;
				fopen_s(&f, args, "w");
				if (f == NULL)
				{
					fprintf(stderr, ERR_FOPEN, args);
					break;
				}
				nn_fwrite(l, a, w12, w23, f);
				fflush(f);
				fclose(f);
				f = NULL;
				break;
			case 'l':
				args = input + 2;
				fopen_s(&f, args, "r");
				if (f == NULL)
				{
					fprintf(stderr, ERR_FOPEN, args);
					break;
				}
				nn_fread(&l, &a, w12, w23, f);
				fclose(f);
				f = NULL;
				break;
			case 'p':
				nn_fprint(l, a, w12, w23, stdout);
				fflush(stdout);
				break;
			case 'r':
				args = input + 2;
				sscanf_s(args, "%lf %lf", &(x[0]), &(x[1]));
				nn_activate(x, w12, w23, l, o1, o2, &oa);
				printf("nn(x1 = %lf, x2 = %lf) = %lf\n", x[0], x[1], oa);
				break;
			case 'u':
				puts(USAGE_MSG);
				break;
			case 'e':
				free(input);
				return 0;
		}

		free(input);
	}
}

static char * finput_al(FILE * stream)
{
	size_t buff_len = INIT_LEN, char_count = 0;
	char * buff = (char *)malloc(buff_len * sizeof(char));
	for (char c; (c = getc(stream)) != EOI;)
	{
		if (ferror(stream))
		{
			fputs("Error while reading input from stream.", stderr);
			abort();
		}

		if (buff_len - char_count <= MIN_LEFTOVER)
		{
			buff_len *= INCR_COEF;
			buff = (char *)realloc(buff, buff_len * sizeof(char));
		}

		buff[char_count++] = c;
	}

	buff[char_count] = '\0';

	return buff;
}