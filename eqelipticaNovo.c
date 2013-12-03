/*
 * Arquivo: eqelipticaNovo.c 
 * -------------------------
 * Este programa implementa algoritmos diretos
 * e iterativos para resolução de um sistema linear
 * tridiagonal. O problema origem deste sistema
 * linear é a resolução numérica da equação do
 * potencial na barra (Equação de Poisson),
 * com condições de contorno de Dirichlet.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * Constantes
 * ----------
 * MaxElementosX -- número máximo de elementos do vetor x;
 * MaxElemNorma -- número máximo de elementos dos vetores que contêm as normas;
 * MaxCasos -- número máximo de discretizações sucessivas;
 * BordaEsq -- extremidade esquerda da barra;
 * BordaDir -- extremidade direita barra;
 * NumNosIni -- número de nós da primeira discretização;
 * Alfa2 -- constante que multiplica o Laplaciano.
 * Omega -- constante do método SOR
 * 		 -- Para matrizes positivas definidas e tridiagonais,
 * 		 -- a escolha ótima é dada por
 *		 -- 		 		   2
 *		 -- Omega =	------------------------ ,
 *		 --			1 + sqrt(1 - [rho(Tj)]²)
 *		 -- onde rho(Tj) é o raio espectral da matriz Tj,
 *		 -- dada por Tj = D⁻¹(L + U).
 *       -- Ref.: Burden & Faires.
 * Diag --
 * OffDiag --
 * MaxIterations --
 * TOL --
 */

#define MaxElementosX 5000 
#define MaxElemNorma 100
#define MaxCasos 7
#define BordaEsq 0.0
#define BordaDir 2.0
#define NumNosIni 8
#define Alfa2 1.0
#define Omega 1.17
#define Diag 2.0
#define OffDiag -1.0
#define MaxIterations 520000
#define TOL 1.0e-6

/* Protótipos das funções */

void MostraInstrucoes(void);
double f(double x);
double g(double x);
void DiscretizaMalha(double x[], int numPontos, double deltax);
double IntervaloDiscr(int numPontos);
void Thomas(int n, double diagBaixo[], double diagPrinc[],
 			double diagCima[], double v[], double u[]);
// Os argumentos de GaussSeidel não incluem os vetores com os
// elementos das diagonais pois diferentemente da função
// Thomas, esses vetores não são usados. Usar esses vetores
// apenas internamente, na função Thomas.
void GaussSeidel(int n, double deltax, double u[], double v[]);
void GaussSeidelRB(int n, double deltax, double u[], double v[]);
void Jacobi(int n, double deltax, double u[], double v[]);
void JacobiParal(int n, double deltax, double u[], double v[]);
void SOR(int n, double deltax, double u[], double v[]);
void SORRB(int n, double deltax, double u[], double v[]);
void CalculaUExata(double uExata[], double x[], int n);
void MatrizTridiag(int n, double diagBaixo[], double diagPrinc[], double diagCima[]);
void ParteDireita(int n, double v[], double deltax, double x[], double uExata[]);
double NormaInfinito(double u[], double uExata[], int n);
double NormaDois(double u[], double uExata[], int n, double deltax);
void TabelaConv(FILE *fp, int caso, int n, double norma2[], double normaInf[]);

/* Programa principal */

main()
{
    int i, n, caso;
    double deltax; 
    double x[MaxElementosX], u[MaxElementosX], uExata[MaxElementosX];
    double diagBaixo[MaxElementosX], diagPrinc[MaxElementosX], diagCima[MaxElementosX], v[MaxElementosX];
    double norma2[MaxElemNorma], normaInf[MaxElemNorma];
    FILE *fp;

    fp = fopen("RazaoConv.txt", (fp == NULL ? "a" : "w"));
    for (caso = 0; caso < 6; caso++) { //debug: voltar com "caso < MaxCasos" OK!
		n = NumNosIni * pow(2, caso);
		deltax = IntervaloDiscr(n);
		DiscretizaMalha(x, n, deltax);
		CalculaUExata(uExata, x, n);
		MatrizTridiag(n, diagBaixo, diagPrinc, diagCima);
		ParteDireita(n, v, deltax, x, uExata);
		//Thomas(n - 1, diagBaixo, diagPrinc, diagCima, v, u);
		//GaussSeidel(n, deltax, u, v);
		//GaussSeidelRB(n, deltax, u, v);
		Jacobi(n, deltax, u, v);
		//JacobiParal(n, deltax, u, v);
		//SOR(n, deltax, u, v);
		//SORRB(n, deltax, u, v);
		normaInf[caso] = NormaInfinito(u, uExata, n);
		norma2[caso] = NormaDois(u, uExata, n, deltax);
		printf("norma2[%d] = %e\n", caso, norma2[caso]); //debug
		TabelaConv(fp, caso, n, norma2, normaInf);
    }
    fclose(fp);

}

/* Função: MostraInstrucoes
 * Uso: MostraInstrucoes();
 * ------------------------
 * Este procedimento imprime instruções para o usuário.
 */

void MostraInstrucoes(void)
{
    printf("Este programa gera uma tabela com as aproximações\n");
    printf("do modelo da Equação do potencial numa barra.\n\n");
}

/* Função: f
 * Uso: termoForcante = f(x);
 * --------------------------
 * Esta função retorna o termo forçante da
 * Equação de Poisson.
 */

double f(double x)
{
	return (- 60 * pow(x, 2));
}

/* Função: g
 * Uso: condContorno = g(x);
 * -------------------------
 * Esta função retorna as condições de contorno da Equação de Poisson.
 */

double g(double x)
{
	return (5 * pow(x, 4) - 2);
}

/*
 * Função: CalculaUExata
 * Uso: CalculaUExata(uExata, x, n);
 * ---------------------------------
 * Esta função preenche o vetor uExata com os valores
 * calculados da solução manufaturada.
 */

void CalculaUExata(double uExata[], double x[], int numPontos)
{
    int i;

    for (i = 0; i <= numPontos; i++) {
		uExata[i] = 5 * pow(x[i], 4) - 2;
    }
}

/* Função: DiscretizaMalha
 * Uso: DiscretizaMalha(x, deltax);
 * --------------------------------
 * Esta função calcula a discretização da barra preenchendo
 * os valores no vetor x.
 */

void DiscretizaMalha(double x[], int numPontos, double deltax)
{
    int i;

    x[0] = BordaEsq;
    x[numPontos] = BordaDir;
    for (i = 1; i < numPontos; i++) {
		x[i] = BordaEsq + i*deltax;
    }
}


/* Função: IntervaloDiscr
 * Uso: deltaX = IntervaloDiscr(numPontos);
 * ----------------------------------------
 * Esta função calcula o intervalo de discretização.
 */

double IntervaloDiscr(int numPontos)
{
    return ( (BordaDir - BordaEsq)/numPontos );
}

/* Função: Thomas
 * Uso: Thomas(nEq, diagBaixo, diagPrinc, diagCima, v, u);
 * ---------------------------------------------------------------
 * Esta função implementa o algoritmo de Thomas para a
 * resolução de um sistema linear Au = b.
 * Fonte: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm.
 * Página visitada em 29/11/2012.
 */

void Thomas(int nEq, double diagBaixo[], double diagPrinc[], double diagCima[], double v[], double u[])
{
   /*
    * nEq - número de equações
    * diagBaixo - sub-diagonal (diagonal abaixo da diagonal principal) -- indexada de 2..nEq
    * diagPrinc - the main diagonal
    * diagCima - sup-diagonal (diagonal acima da diagonal principal) -- indexada de 1..nEq-1
    * v - parte direita
    * u - resposta
    */
    int i;
    double m;

    for (i = 2; i <= nEq; i++) {
		m = diagBaixo[i] / diagPrinc[i-1];
		diagPrinc[i] = diagPrinc[i] - m * diagCima[i-1];
		v[i] = v[i] - m * v[i-1];
    }
    u[nEq] = v[nEq] / diagPrinc[nEq];
    for (i = nEq - 1; i >= 1; i--) {
		u[i] = (v[i] - diagCima[i] * u[i+1]) / diagPrinc[i];
    }
}

/*
 * Função: GaussSeidel
 * Uso: ...;
 * ---------
 * ...
 */

void GaussSeidel(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX], uAux[MaxElementosX]; //uAux for debug

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) {
		u[1] = (v[1] - OffDiag * u0[2]) / Diag;
		for (i = 2; i < n - 1; i++) {
			u[i] = (-OffDiag * u[i - 1] - OffDiag * u0[i + 1] + v[i]) / Diag;
		}
		u[n - 1] = (-OffDiag * u[n - 2] + v[n - 1]) / Diag;
		/*printf("u\tu0\n"); //debug
		for (i = 1; i < n; i++) {
			uAux[i] = 0;
			printf("%f\t%f\n", u[i], u0[i]);
		}*/ //debug
		if (NormaDois(u, u0, n, 1) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}

/*
 * Função: GaussSeidelRB
 * Uso: ...;
 * ---------
 * ...
 * Ver http://ocw.mit.edu/courses/mathematics/18-086-mathematical-methods-for-engineers-ii-spring-2006/readings/am62.pdf (página visitada na 1a. quinzena setembro)
 */

void GaussSeidelRB(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX];

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) {
		u[1] = (v[1] - OffDiag * u0[2]) / Diag;
		for (i = 2; i <= (n % 2 == 0 ? n - 2 : n - 3); i += 2) {
			u[i] = (-OffDiag * u0[i - 1] - OffDiag * u0[i + 1] + v[i]) / Diag;
		}
		for (i = 1; i <= (n % 2 == 0 ? n - 3 : n - 2); i += 2) {
			u[i] = (-OffDiag * u[i - 1] - OffDiag * u[i + 1] + v[i]) / Diag;
		}
		u[n - 1] = (-OffDiag * u[n - 2] + v[n - 1]) / Diag;
		if (NormaDois(u, u0, n, 1) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}

/*
 * Função: Jacobi
 * Uso: Jacobi(n, deltax, u, v);
 * -----------------------------
 * ...
 */

void Jacobi(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX];

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) { //debug: voltar com (k <= MaxIterations)
		u[1] = (v[1] - OffDiag * u0[2]) / Diag;
		for (i = 2; i < n - 1; i++) {
			u[i] = (-OffDiag * u0[i - 1] - OffDiag * u0[i + 1] + v[i]) / Diag;
			//printf("u[%d] = %f, u0[.] = %f\n", i, u[i], u0[i]); //debug
		}
		u[n - 1] = (-OffDiag * u0[n - 2] + v[n - 1]) / Diag;

		/*for (i = 1; i < n; i++) {
			printf("u[%d] = %f, u0[.] = %f\n", i, u[i], u0[i]); //debug
		}*/ //debug

		if (NormaDois(u, u0, n, 1) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		//printf("%d\n", k); //debug
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}

/*
 * Função: JacobiParal
 * Uso: JacobiParal(n, deltax, u, v);
 * -----------------------------
 * ...
 */

void JacobiParal(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX], uAux[MaxElementosX]; //uAux for debug

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) {
		u[1] = (v[1] - OffDiag * u0[2]) / Diag;
		for (i = 2; i <= (n % 2 == 0 ? n - 2 : n - 3); i += 2) {
			u[i] = (-OffDiag * u0[i - 1] - OffDiag * u0[i + 1] + v[i]) / Diag;
		}
		for (i = 1; i <= (n % 2 == 0 ? n - 3 : n - 2); i += 2) {
			u[i] = (-OffDiag * u0[i - 1] - OffDiag * u0[i + 1] + v[i]) / Diag;
		}
		u[n - 1] = (-OffDiag * u0[n - 2] + v[n - 1]) / Diag;
		/*printf("u\tu0\n"); //debug
		for (i = 1; i < n; i++) {
			uAux[i] = 0;
			printf("%f\t%f\n", u[i], u0[i]);
		}*/ //debug
		if (NormaDois(u, u0, n, 1) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}

/*
 * Função: SOR
 * Uso: SOR(n, deltax, u, v);
 * -----------------------------
 * ...
 */

void SOR(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX], uAux[MaxElementosX]; //uAux for debug

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) {
		u[1] = (1 - Omega)*u0[1] + Omega*(v[1] - OffDiag*u0[2]) / Diag;
		for (i = 2; i < n - 1; i++) {
			u[i] = (1 - Omega)*u0[i] + Omega*(-OffDiag*u[i - 1] - OffDiag*u0[i + 1] + v[i]) / Diag;
		}
		u[n - 1] = (1 - Omega)*u0[n - 1] + Omega*(-OffDiag * u[n - 2] + v[n - 1]) / Diag;
		/*printf("u\tu0\n"); //debug
		for (i = 1; i < n; i++) {
			uAux[i] = 0;
			printf("%f\t%f\n", u[i], u0[i]);
		}*/ //debug
		if (NormaDois(u, u0, n, 1) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}

/*
 * Função: SORRB
 * Uso: SOR(n, deltax, u, v);
 * -----------------------------
 * ...
 * A constante Omega definida acima, corresponde ao valor
 * aproximado relativo ao cálculo explicitado quando de sua
 * definição, para n = 3. 
 */

void SORRB(int n, double deltax, double u[], double v[])
{
	int i, k;
	double u0[MaxElementosX], uAux[MaxElementosX]; //uAux for debug

	k = 0;
	for (i = 1; i < n; i++) {
		u0[i] = 0;
	}
	while (k <= MaxIterations) {
		u[1] = (1 - Omega)*u0[1] + Omega*(v[1] - OffDiag*u0[2]) / Diag;
		for (i = 2; i <= (n % 2 == 0 ? n - 2 : n - 3); i += 2) {
			u[i] = (1 - Omega)*u0[i] + Omega*(-OffDiag*u0[i - 1] - OffDiag*u0[i + 1] + v[i]) / Diag;
		}
		for (i = 1; i <= (n % 2 == 0 ? n - 3 : n - 2); i += 2) {
			u[i] = (-OffDiag * u[i - 1] - OffDiag * u[i + 1] + v[i]) / Diag;
			u[i] = (1 - Omega)*u0[i] + Omega*(-OffDiag*u[i - 1] - OffDiag*u[i + 1] + v[i]) / Diag;
		}
		u[n - 1] = (1 - Omega)*u0[n - 1] + Omega*(-OffDiag * u[n - 2] + v[n - 1]) / Diag;
		/*printf("u\tu0\n"); //debug
		for (i = 1; i < n; i++) {
			uAux[i] = 0;
			printf("%f\t%f\n", u[i], u0[i]);
		}*/ //debug
		if (NormaDois(u, u0, n, deltax) < TOL) {
			printf("Iterações: %d\n", k); //debug
			return; //Aproximação gerada com sucesso
		}
		k++;
		for (i = 1; i < n; i++) {
			u0[i] = u[i];
		}
	}
	printf("Número de iterações máximo excedido.\n");
}
/*
 * Função: MatrizTridiag
 * Uso: MatrizTridiag(numPontos, diagBaixo, diagPrinc, diagCima);
 * --------------------------------------------------------------
 * Esta função preenche os vetores diagBaixo, diagCima, diagPrinc com
 * os valores das três diagonais não-nulas da Matriz do sistema linear
 * resultante da discretização da equação de Poisson na barra por 
 * diferenças finitas. 
 */

void MatrizTridiag(int numPontos, double diagBaixo[], double diagPrinc[], double diagCima[])
{
    int i;

    for(i = 1; i < numPontos; i++) {
		diagBaixo[i] = -1;
		diagCima[i] = -1;
		diagPrinc[i] = 2;
    }
}

/*
 * Função: ParteDireita
 * Uso: ParteDireita(n, v, Alfa2, uExata);
 * ---------------------------------------
 * Esta função preenche o o vetor v com os valores do lado
 * direito do sistema linear obtido através da discretização da
 * Equação de Poisson, por diferenças finitas.
 */

void ParteDireita(int n, double v[], double deltax, double x[], double uExata[])
{
    int i;

    for(i = 2; i < n - 1; i++) {
		v[i] = pow(deltax, 2) * f(x[i]);
    }
    v[1] = pow(deltax, 2) * f(x[1]) + g(BordaEsq);
    v[n - 1] = pow(deltax, 2) * f(x[n - 1]) + g(BordaDir);
}

/*
 * Função: NormaInfinito
 * Uso: NormaInfinito(u, uExata, n);
 * ---------------------------------
 * Esta função calcula a norma infinito da difernça entre os
 * vetores u e uExata.
 */

double NormaInfinito(double u[], double uExata[], int n)
{
    int i;
    double normaInf;

    normaInf = fabs(u[1] - uExata[1]);
    for (i = 2; i <= n - 1; i++) {
		if (fabs(u[i] - uExata[i]) > normaInf) {
			normaInf = fabs(u[i] - uExata[i]);
		}
    }
    return normaInf;
}

/*
 * Função: NormaDois
 * Uso: NormaDois(u, uExata, n, deltax);
 * -------------------------------------
 * Esta função calcula a norma 2 da diferença entre os vetores
 * u e uExata.
 */

double NormaDois(double u[], double uExata[], int n, double deltax)
{
    int i;
    double normaDois;

    normaDois = 0;
    for (i = 1; i < n; i++) { //debug: voltar com "i = 1", "i < n" (OK!)
		normaDois += pow(u[i] - uExata[i], 2) * deltax;
    }
	//printf("%e\n", sqrt(normaDois)); //debug (OK!)
	return (sqrt(normaDois));
}

/*
 * Função: TabelaConv
 * Uso: TabelaConv(fp, caso, n, norma2, normaInf);
 * -----------------------------------------------
 * Este procedimento escreve o arquivo fp, com as normas infinito e
 * norma 2 para os sucessivos casos, representados pelas sucessivas
 * divisões dos nós da barra por dois. A tabela gerada contém
 * as razões de convergência conforme se refina a malha de pontos.
 * Na primeira linha da tabela altere a solução manufaturada escrita
 * com a solução manufaturada que você escolheu.
 */

void TabelaConv(FILE *fp, int caso, int n, double norma2[], double normaInf[])
{
    if (caso == 0) {
		fprintf(fp, "Solução manufaturada: u(x) = 5x^4 - 2\n");
		fprintf(fp, "alfa2 = %f\n", Alfa2);
		fprintf(fp, "x0 = %f, xL = %f\n", BordaEsq, BordaDir);
		fprintf(fp, "Caso |   n   |  Norma 2  | Razão conv. | Norma inf. | Razão conv. |\n");
		fprintf(fp, "-----|-------|-----------|-------------|------------|-------------|\n");
		fprintf(fp, " %2d  |%6d |  %8.6f |      -      |  %8.6f  |      -      |\n", caso, n,
		norma2[caso], normaInf[caso]);
    } else {
		fprintf(fp, " %2d  |%6d |  %8.6f |  %8.6f   |  %8.6f  |  %8.6f   |\n", caso, n,
		norma2[caso], norma2[caso - 1] / norma2[caso], normaInf[caso], normaInf[caso - 1] / normaInf[caso]);
    }
}

