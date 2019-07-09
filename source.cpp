
#include "mvector.h"
#include "mmatrix.h"
#include <iomanip>  
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <cassert>


// Set up a "random device" that generates a new random number each time the program is run
std::random_device rand_dev;

// Set up a pseudo-random number generater "rnd", seeded with a random number
std::mt19937 rnd(rand_dev());




//Operator Overloads

// MMatrix * MVector
MVector operator*(const MMatrix &m, const MVector &v)
{
	assert(m.Cols() == v.size());

	MVector r(m.Rows());

	for (int i = 0; i < m.Rows(); i++)
	{
		for (int j = 0; j < m.Cols(); j++)
		{
			r[i] += m(i, j)*v[j];
		}
	}
	return r;
}

//Plus on matrices
MMatrix operator+(const MMatrix &m, const MMatrix &n)
{
	assert(m.Cols() == n.Cols());
	assert(m.Rows() == n.Rows());
	MMatrix outMat(m.Rows(), m.Cols());

	for (int i = 0; i < m.Rows(); i++)
	{
		for (int j = 0; j < m.Cols(); j++)
		{
			outMat(i, j) = m(i, j)+n(i, j);
		}
	}
	return outMat;
}

//Componentwise Mult. on matrices
MMatrix ComponentwiseMultiplication(const MMatrix &m, const MMatrix &n)
{
	assert(m.Cols() == n.Cols());
	assert(m.Rows() == n.Rows());
	MMatrix outMat(m.Rows(),m.Cols());
	for (int i = 0; i < m.Rows(); i++)
	{
		for (int j = 0; j < m.Cols(); j++)
		{
			outMat(i, j) = m(i, j)*n(i, j);
		}
	}
	return outMat;
}

//Componentwise mult. on vectors
MVector ComponentwiseMultiplication(const MVector &m, const MVector &n)
{
	assert(m.size() == n.size());
	
	MVector outvect(m.size());
	for (int i = 0; i < m.size(); i++)
	{
		outvect[i] = m[i]*n[i];
		
	}
	return outvect;
}

// transpose(MMatrix) * MVector
MVector TransposeTimes(const MMatrix &m, const MVector &v)
{
	assert(m.Rows() == v.size());

	MVector r(m.Cols());

	for (int i = 0; i < m.Cols(); i++)
	{
		for (int j = 0; j < m.Rows(); j++)
		{
			r[i] += m(j, i)*v[j];
		}
	}
	return r;
}

// MVector + MVector
MVector operator+(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i = 0; i < lhs.size(); i++)
		r[i] += rhs[i];

	return r;
}

// MVector - MVector
MVector operator-(const MVector &lhs, const MVector &rhs)
{
	assert(lhs.size() == rhs.size());

	MVector r(lhs);
	for (int i = 0; i < lhs.size(); i++)
		r[i] -= rhs[i];

	return r;
}

// MMatrix = MVector <outer product> MVector
// M = a <outer product> b
MMatrix OuterProduct(const MVector &a, const MVector &b)
{
	MMatrix m(a.size(), b.size());
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++)
		{
			m(i, j) = a[i] * b[j];
		}
	}
	return m;
}

// Hadamard product
MVector operator*(const MVector &a, const MVector &b)
{
	assert(a.size() == b.size());

	MVector r(a.size());
	for (int i = 0; i < a.size(); i++)
		r[i] = a[i] * b[i];
	return r;
}

// double * MMatrix
MMatrix operator*(double d, const MMatrix &m)
{
	MMatrix r(m);
	for (int i = 0; i < m.Rows(); i++)
		for (int j = 0; j < m.Cols(); j++)
			r(i, j) *= d;

	return r;
}

// double * MVector
MVector operator*(double d, const MVector &v)
{
	MVector r(v);
	for (int i = 0; i < v.size(); i++)
		r[i] *= d;

	return r;
}

// MVector -= MVector
MVector operator-=(MVector &v1, const MVector &v)
{
	assert(v1.size() == v.size());

	MVector r(v1);
	for (int i = 0; i < v1.size(); i++)
		v1[i] -= v[i];

	return r;
}

// MMatrix -= MMatrix
MMatrix operator-=(MMatrix &m1, const MMatrix &m2)
{
	assert(m1.Rows() == m2.Rows() && m1.Cols() == m2.Cols());

	for (int i = 0; i < m1.Rows(); i++)
		for (int j = 0; j < m1.Cols(); j++)
			m1(i, j) -= m2(i, j);

	return m1;
}

// Output function for MVector
inline std::ostream &operator<<(std::ostream &os, const MVector &rhs)
{
	std::size_t n = rhs.size();
	os << "(";
	for (std::size_t i = 0; i < n; i++)
	{
		os << rhs[i];
		if (i != (n - 1)) os << ", ";
	}
	os << ")";
	return os;
}

// Output function for MMatrix
inline std::ostream &operator<<(std::ostream &os, const MMatrix &a)
{
	int c = a.Cols(), r = a.Rows();
	for (int i = 0; i < r; i++)
	{
		os << "(";
		for (int j = 0; j < c; j++)
		{
			os.width(10);
			os << a(i, j);
			os << ((j == c - 1) ? ')' : ',');
		}
		os << "\n";
	}
	return os;
}



// Functions that provide sets of training data

// Generate 16 points of training data in the pattern illustrated in the project description
void GetTestData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	x = { {0.125,.175}, {0.375,0.3125}, {0.05,0.675}, {0.3,0.025}, {0.15,0.3}, {0.25,0.5}, {0.2,0.95}, {0.15, 0.85},
		 {0.75, 0.5}, {0.95, 0.075}, {0.4875, 0.2}, {0.725,0.25}, {0.9,0.875}, {0.5,0.8}, {0.25,0.75}, {0.5,0.5} };

	y = { {1},{1},{1},{1},{1},{1},{1},{1},
		 {-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1} };
}

// Generate 1000 points of test data in a checkerboard pattern
void GetCheckerboardData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	for (int i = 0; i < 1000; i++)
	{
		x[i] = { lr() / static_cast<double>(lr.max()),lr() / static_cast<double>(lr.max()) };
		double r = sin(x[i][0] * 12.5)*sin(x[i][1] * 12.5);
		y[i][0] = (r > 0) ? 1 : -1;
	}
}


// Generate 1000 points of test data in a spiral pattern
void GetSpiralData(std::vector<MVector> &x, std::vector<MVector> &y)
{
	std::mt19937 lr;
	x = std::vector<MVector>(1000, MVector(2));
	y = std::vector<MVector>(1000, MVector(1));

	double twopi = 8.0*atan(1.0);
	for (int i = 0; i < 1000; i++)
	{
		x[i] = { lr() / static_cast<double>(lr.max()),lr() / static_cast<double>(lr.max()) };
		double xv = x[i][0] - 0.5, yv = x[i][1] - 0.5;
		double ang = atan2(yv, xv) + twopi;
		double rad = sqrt(xv*xv + yv * yv);

		double r = fmod(ang + rad * 20, twopi);
		y[i][0] = (r < 0.5*twopi) ? 1 : -1;
	}
}

// Save the the training data in x and y to a new file, with the filename given by "filename"
// Returns true if the file was saved succesfully
bool ExportTrainingData(const std::vector<MVector> &x, const std::vector<MVector> &y,
	std::string filename)
{
	// Check that the training vectors are the same size
	assert(x.size() == y.size());

	// Open a file with the specified name.
	std::ofstream f(filename);

	// Return false, indicating failure, if file did not open
	if (!f)
	{
		return false;
	}

	// Loop over each training datum
	for (unsigned i = 0; i < x.size(); i++)
	{
		// Check that the output for this point is a scalar
		assert(y[i].size() == 1);

		// Output components of x[i]
		for (int j = 0; j < x[i].size(); j++)
		{
			f << x[i][j] << " ";
		}

		// Output only component of y[i]
		f << y[i][0] << " " << std::endl;
	}
	f.close();

	if (f) return true;
	else return false;
}





// Neural network class

class Network
{
public:

	// sets up vectors of MVectors and MMatrices for
	// weights, biases, weighted inputs, activations and errors
	// The parameter nneurons_ is a vector defining the number of neurons at each layer.
	// For example:
	//   Network({2,1}) has two input neurons, no hidden layers, one output neuron
	//
	//   Network({2,3,3,1}) has two input neurons, two hidden layers of three neurons
	//                      each, and one output neuron
	Network(std::vector<unsigned> nneurons_)
	{
		nneurons = nneurons_;
		nLayers = nneurons.size();
		weights = std::vector<MMatrix>(nLayers);
		biases = std::vector<MVector>(nLayers);
		errors = std::vector<MVector>(nLayers);
		activations = std::vector<MVector>(nLayers);
		inputs = std::vector<MVector>(nLayers);
		// Create activations vector for input layer 0
		activations[0] = MVector(nneurons[0]);

		// Other vectors initialised for second and subsequent layers
		for (unsigned i = 1; i < nLayers; i++)
		{
			weights[i] = MMatrix(nneurons[i], nneurons[i - 1]);
			biases[i] = MVector(nneurons[i]);
			inputs[i] = MVector(nneurons[i]);
			errors[i] = MVector(nneurons[i]);
			activations[i] = MVector(nneurons[i]);
		}


	}

	// Return the number of input neurons
	unsigned NInputNeurons() const
	{
		return nneurons[0];
	}

	// Return the number of output neurons
	unsigned NOutputNeurons() const
	{
		return nneurons[nLayers - 1];
	}

	// Evaluate the network for an input x and return the activations of the output layer
	MVector Evaluate(const MVector &x)
	{
		// Call FeedForward(x) to evaluate the network for an input vector x
		FeedForward(x);

		// Return the activations of the output layer
		return activations[nLayers - 1];
	}


	int Train(const std::vector<MVector> x, const std::vector<MVector> y,
		double initsd, double learningRate, double costThreshold, int maxIterations)
	{
		// Check that there are the same number of training data inputs as outputs
		assert(x.size() == y.size());
		
		
		InitialiseWeightsAndBiases(initsd);
		// initialise the weights and biases with the standard deviation "initsd"

		std::ofstream f;
		f.open("cost0.001.txt");

		for (int iter = 1; iter <= maxIterations; iter++)
		{
			// Choose a random training data point i in {0, 1, 2, ..., N}
			int i = rnd() % x.size();
			
			
			// run the feed-forward algorithm
			FeedForward(x[i]);
			// run the back-propagation algorithm
			BackPropagateError(y[i]);
			// update the weights and biases using stochastic gradient
			//               with learning rate "learningRate"
			
			

			UpdateWeightsAndBiases(learningRate);
			// Every so often, perform step 7 and show an update on how the cost function has decreased
			
			if ((!(iter % 1000)) || iter == maxIterations)
			{
				double cost = TotalCost(x, y);
				//calculate the total cost
				
				std::cout << "" << iter << "  " << cost << std::endl;
				f << iter << "  " << cost << std::endl;
				// display the iteration number and total cost to the screen
				
				if (cost < costThreshold)
				{
					
					return iter;
				}
				//  return from this method with a value of true,
				//              indicating success, if this cost is less than "costThreshold".
			}

		} 
		

		// return "false", indicating that the training did not succeed.
		return -1;
	}

	void out(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		for (int i = 0; i < x.size(); i++)
		{
			std::cout << x[i] << " " << y[i] << " " << activations[activations.size() - 1][i] << std::endl;
		}
	}

	// For a neural network with two inputs x=(x1, x2) and one output y,
	// loop over (x1, x2) for a grid of points in [0, 1]x[0, 1]
	// and save the value of the network output y evaluated at these points
	// to a file. Returns true if the file was saved successfully.
	bool ExportOutput(std::string filename)
	{
		// Check that the network has the right number of inputs and outputs
		assert(NInputNeurons() == 2 && NOutputNeurons() == 1);

		// Open a file with the specified name.
		std::ofstream f(filename);

		// Return false, indicating failure, if file did not open
		if (!f)
		{
			return false;
		}

		// generate a matrix of 250x250 output data points
		for (int i = 0; i <= 250; i++)
		{
			for (int j = 0; j <= 250; j++)
			{
				MVector out = Evaluate({ i / 250.0, j / 250.0 });
				f << out[0] << " ";
			}
			f << std::endl;
		}
		f.close();

		if (f) return true;
		else return false;
	}


	static bool Test();
	
	private:
	// Return the activation function sigma
	double Sigma(double z)
	{
		return ((exp(2 * z) - 1) / (exp(2 * z) + 1));
	}

	// Return the derivative of the activation function
	double SigmaPrime(double z)
	{
		return 1 - pow((exp(2 * z) - 1) / (exp(2 * z) + 1), 2);
	}

	// Loop over all weights and biases in the network and set each
	// term to a random number normally distributed with mean 0 and
	// standard deviation "initsd"
	void InitialiseWeightsAndBiases(double initsd)
	{
		// Make sure the standard deviation supplied is non-negative
		assert(initsd >= 0);

		// Set up a normal distribution with mean zero, standard deviation "initsd"
		// Calling "dist(rnd)" returns a random number drawn from this distribution 
		std::normal_distribution<> dist(0, initsd);


		//for each layer in the network between the second and last
		for (int layeri = 1; layeri < nLayers; layeri++)
		{
			MMatrix weightMatrixi = weights[layeri];
			
			//iterate over each element in the weight matrix
			for (int columni = 0; columni < weightMatrixi.Cols(); columni++)
			{
				for (int rowi = 0; rowi < weightMatrixi.Rows(); rowi++)
				{
					//set according to normal dist
					weights[layeri](rowi, columni) = dist(rnd);
					
				}
			}

			//set vector of biases according to normal dist
			for (int posi = 0; posi < biases[layeri].size(); posi++)
			{
				biases[layeri][posi] = dist(rnd);
			}
			
			
		}


	}

	// Evaluate the feed-forward algorithm, setting weighted inputs and activations
	// at each layer, given an input vector x
	void FeedForward(const MVector &x)
	{
		// Check that the input vector has the same number of elements as the input layer
		assert(x.size() == nneurons[0]);

		
		for (int layeri = 0; layeri < nLayers; layeri++)
		{
			if (layeri == 0)
			{
				activations[layeri] = x;
				
			}
			else
			{
				inputs[layeri] = (weights[layeri] * activations[layeri - 1] + biases[layeri]);
				//perform Sigma function componentwise
				for (int i = 0; i < inputs[layeri].size(); i++) {
					activations[layeri][i] = Sigma(inputs[layeri][i]);
				}
			}
		}
	
	}

	// Evaluate the back-propagation algorithm, setting errors for each layer 
	void BackPropagateError(const MVector &y)
	{
		// Check that the output vector y has the same number of elements as the output layer
		assert(y.size() == nneurons[nLayers - 1]);

		for (int i = nLayers - 1; i > 0; i--)
		{
			//for the final layer, do equation 1.22
			if (i == nLayers - 1)
			{
				//perform sigmaprime componentwise first
				for (int j = 0; j < inputs[i].size(); j++) {

					errors[i][j] = SigmaPrime(inputs[i][j]);
				}
				
				errors[i] = ComponentwiseMultiplication(errors[i], (activations[i] - y));
			}

			//for the other layers, do equation 1.24
			else
			{
				//perform sigmaprime componentwise first
				for (int j = 0; j < inputs[i].size(); j++) {

					errors[i][j] = SigmaPrime(inputs[i][j]);
				}
				
				errors[i] = ComponentwiseMultiplication(errors[i],TransposeTimes(weights[i+1],errors[i+1]));
			}
			
		}

		
	}


	// Apply one iteration of the stochastic gradient iteration with learning rate eta.
	void UpdateWeightsAndBiases(double eta)
	{
		
		// Check that the learning rate is positive
		assert(eta > 0);
		//iterate over all layers, and apply eqs 1.25, 1.26
		for (int i = 1; i < nLayers; i++)
		{
			biases[i] = biases[i] - eta * errors[i];
			weights[i] = weights[i] + -eta * OuterProduct(errors[i],activations[i-1]);
		}
		
	}


	// Return the cost function of the network with respect to a single the desired output y
	// call FeedForward(x) first to evaluate the network output for an input x,
	//       then call this method Cost(y) with the corresponding desired output y
	double Cost(const MVector &y)
	{
		// Check that y has the same number of elements as the network has outputs
		assert(y.size() == nneurons[nLayers - 1]);
		double twoNormSum = 0;
		//2 norm squared over each element of y
		for (int i = 0; i < y.size(); i++)
		{
			twoNormSum += pow(y[i] - activations[nLayers - 1][i], 2);
			//NOTE THAT THIS ONLY WORKS IF YOU CALL FEEDFOWARD FIRST
		}
		return 0.5*twoNormSum;
		
	}

	// Return the total cost C for a set of training data x and desired outputs y
	double TotalCost(const std::vector<MVector> x, const std::vector<MVector> y)
	{
		// Check that there are the same number of inputs as outputs
		assert(x.size() == y.size());
		double totalCost = 0;

		//feed foward all data points and then add the cost to the total cost
		for (int i = 0; i < x.size(); i++)
		{
			FeedForward(x[i]);
			totalCost+=Cost(y[i]);
		}

		//take the average
		return totalCost = (1.0 / x.size())*totalCost;
		
	}

	// Private member data

	std::vector<unsigned> nneurons;
	std::vector<MMatrix> weights;
	std::vector<MVector> biases, errors, activations, inputs;
	unsigned nLayers;

};



bool Network::Test()
{
	
	// This function should return true if all tests pass, or false otherwise

	

	// A example test of FeedForward
	{
		// Make a simple network with two weights and one bias
		Network n({ 2, 1 });

		// Set the values of these by hand
		n.biases[1][0] = 0.5;
		n.weights[1](0, 0) = -0.3;
		n.weights[1](0, 1) = 0.2;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({ 0.3, 0.4 });

		// Display the output value calculated
		std::cout << n.activations[1][0] << std::endl;

		// Correct value is = tanh(0.5 + (-0.3*0.3 + 0.2*0.4))
		//                    = 0.454216432682259...
		// Fail if error in answer is greater than 10^-10:
		if (std::abs(n.activations[1][0] - 0.454216432682259) > 1e-10)
		{
			return false;
		}
		else
		{
			std::cout << "Feed foward working" << std::endl;
		}
		n.BackPropagateError({ 1 });
			std::cout << n.errors[1];
		

	}

	//2D Cost() test
	{
		// Make a two by two network, i.e. 4 weights, 2 biases
		Network n({ 2, 2 });
		
		// Set the values of these by hand
		
		n.biases[1][0] = 0.5;
		n.biases[1][1] = 0.5;
		n.weights[1](0, 0) = 0.5;
		n.weights[1](0, 1) = 0.5;
		n.weights[1](1, 0) = 0.5;
		n.weights[1](1, 1) = 0.5;

		// Call function to be tested with x = (0.3, 0.4)
		n.FeedForward({ 0.5, 0.5 });

		// Display the output value calculated
		std::cout << n.activations[1][0] << std::endl;
		std::cout << n.activations[1][1] << std::endl;
		std::cout << n.Cost({ 1, 1 }) << std::endl;
		//the output is going to be given by (tanh(0.5*0.5+0.5*0.5)+0.5),tanh(0.5*0.5+0.5*0.5)+0.5))
		//which =(0.76159415595,0.76159415595)
		//So Cost=1/2*TwoNormSquaredOf((0.24..., 0.24...))
		//=0.05683734647
		if (std::abs(0.05683734647 - n.Cost({ 1, 1 })) > 1e-10)
		{
			return false;
		}
		
		std::cout << "Cost working" << std::endl;
		if (std::abs(0.05683734647 - n.Cost({ 1, 1 })) > 1e-10)
		{
			return false;
		}
		
		if (std::abs(0.0483189933965 - n.TotalCost({ { 0.5, 0.5 }, { 0.6, 0.6 } }, { { 1, 1 }, { 1, 1 } })) > 1e-10)
		{
			return false;
		}

		
		std::cout <<"total cost working"<< std::endl;

	}

	
	//'Test' of InitialiseWeightsandBiases
	{
		Network n({ 2, 1 });
		n.InitialiseWeightsAndBiases(1);
	}

	//Test Sigma functions
	{
		//Have to create a network since Sigma is a non static member function
		Network n({ 1,1 });

		//Test against precalculated values
		if (std::abs(0.76159415595 - n.Sigma(1)) > 1e-10)
		{
			return false;
		}
		if (std::abs(0.41997434161 - n.SigmaPrime(1)) > 1e-10)
		{
			return false;
		}

		std::cout << "Sigma working" << std::endl;
	}

	//Test Backpropagaterror
	{
		Network n({ 2, 2, 2, 1 });
		n.biases[3][0] = 0.5;
		n.biases[2][0] = 0.41;
		n.biases[2][1] = 0.42;
		n.biases[1][0] = 0.43;
		n.biases[1][1] = 0.44;
		n.weights[3](0, 0) = -0.31;
		n.weights[3](0, 1) = 0.22;
		n.weights[2](0, 0) = -0.33;
		n.weights[2](0, 1) = 0.24;
		n.weights[2](1, 0) = -0.35;
		n.weights[2](1, 1) = 0.26;
		n.weights[1](0, 0) = -0.37;
		n.weights[1](0, 1) = 0.28;
		n.weights[1](1, 0) = -0.31;
		n.weights[1](1, 1) = 0.22;

		n.FeedForward({ 0.3, 0.4 });
		n.BackPropagateError({ 1 });

		if (std::abs(n.errors[3][0] - -0.454206850960898) > 1e-10) return false;
		if (std::abs(n.errors[2][0] - 0.122788311005948) > 1e-10) return false;
		if (std::abs(n.errors[2][1] - -0.0865071609138907) > 1e-10) return false;
		if (std::abs(n.errors[1][0] - -0.00855297771367315) > 1e-10) return false;
		if (std::abs(n.errors[1][1] - 0.00580735128811967) > 1e-10) return false;


		std::cout << "BackPropagate Working" << std::endl;

		//Test update weights and biases
		n.UpdateWeightsAndBiases(0.1);
		
		for (int i=0;i<n.weights.size();i++)
		{
			std::cout << std::setprecision(14) << n.weights[i] << std::endl;
		}
		for (int i = 0; i < n.biases.size(); i++)
		{
			std::cout<<std::setprecision(14) << n.biases[i] << std::endl;
		}
		if (std::abs(n.weights[1](0,0) + 0.369743410669) > 1e-10) return false;
		if (std::abs(n.biases[3][0] - 0.545420685096) > 1e-10) return false;
	}

	

	return true;
}


// Main function and example use of the Network class

void ClassifyTestData()
{
	// Create a network with two input neurons, two hidden layers of three neurons, and one output neuron
	Network n({ 2, 4, 4, 1 });

	// Get some data to train the network
	std::vector<MVector> x, y;
	GetTestData(x, y);
	ExportTrainingData(x, y, "test_points.txt");
	// Train network on training inputs x and outputs y
	// Numerical parameters are:
	//  initial weight and bias standard deviation = 0.1
	//  learning rate = 0.1
	//  cost threshold = 1e-4
	//  maximum number of iterations = 10000


		int trainingSucceeded = n.Train(x, y, 0.1, 0.1, 1e-4, 1000000000);



	// If training failed, report this
	if (trainingSucceeded==-1)
	{
		std::cout << "Failed to converge to desired tolerance." << std::endl;
	}

	// Generate some output files for plotting
	
	n.ExportOutput("test_contour.txt");
}

int main()
{
	// Call the test function	
	bool testsPassed = Network::Test();

	// If tests did not pass, something is wrong; end program now
	if (!testsPassed)
	{
		std::cout << "A test failed." << std::endl;
		return 1;
	}

	// Tests passed, so run our example program.
	ClassifyTestData();

	return 0;
}
