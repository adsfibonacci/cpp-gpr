#include <iostream>
#include <cmath>
#include <functional>
#include <eigen3/Eigen/Dense>
#include <utility>






using std::cout, std::endl, std::pair, std::function, std::make_pair, std::bind;
using Eigen::Dynamic, Eigen::Matrix, Eigen::Map;
float gaussian_kernel(const unsigned int dim,
					  const float *d1,
					  const float *d2,
					  const float sigma = 1,
					  const float length = 1) {
  float totals = 0;
  for(unsigned int i = 0; i < dim; ++i) {
	totals += pow(*(d1 + i) - *(d2 + i), 2);
  }
  return pow(sigma, 2) * exp(-totals / ( 2*pow(length,2)));
}
typedef function<float(const float*, const float*)> KernType;
function<float(const float*, const float*)> kern(unsigned int dim, float std, float length) {
  return bind(gaussian_kernel, dim, std::placeholders::_1, std::placeholders::_2, std, length);
}
Matrix<float, Dynamic, Dynamic> covariance_matrix(
														 const unsigned int dim,
														 const unsigned int count,
														 const float *data,
														 const KernType ker
														 ) {
  Matrix<float, Dynamic, Dynamic> cov_mat(count, count);
  for(unsigned int i = 0; i < count; ++i) {
  	for(unsigned int j = 0; j <= i; ++j) {
  	  cov_mat(i, j) = ker(data + i * dim, data + j * dim);
  	  cov_mat(j, i) = cov_mat(i, j);
  	}
  }
  return cov_mat;
}
Matrix<float, Dynamic, 1> covariance_vector(
														  const unsigned int dim,
														  const unsigned int count,
														  const float *data,
														  const float *point,
														  const function<float(const float*, const float*)> ker
														  ) {
  Matrix<float, Dynamic, 1> cov_vec(count);
  for(unsigned int i = 0; i < count; ++i) {
	cov_vec(i) = ker(data + i * dim, point);
  }
  return cov_vec;
}
pair<float, float> predict(const unsigned int dim,
								const Matrix<float, Dynamic, 1>& response,
								const Matrix<float, Dynamic, Dynamic>& cov_mat,
								const Matrix<float, Dynamic, 1>& cov_vec
								) {
  return make_pair(cov_vec.transpose() * cov_mat.inverse() * response,
						1 - cov_vec.transpose() * cov_mat.inverse() * cov_vec);
}
pair<float, float> gpr_model(const unsigned int dim, const unsigned int count, const float *data,
								  const float *response,
								  const float *point, KernType ker) {
  Matrix<float, Dynamic, Dynamic> cov_mat(count, count);
  Matrix<float, Dynamic, 1> cov_vec(count);
  Map<const Matrix<float, Dynamic, 1>> res(response, count);
  cov_mat = covariance_matrix(dim, count, data, ker);
  cov_vec = covariance_vector(dim, count, data, point, ker);
  return predict(dim, res, cov_mat, cov_vec);
}
int main() {
  /* Define 2 dimensions and 5 datapoints*/
  const unsigned int dim = 2;
  const unsigned int count = 5;
  
  /* Define the datapoints */ 
  float data[count][dim] = {
    {.12, .55},
    {.80, .45},
    {.17, .23},
    {.52, .38},
    {.51, .62}
  };
  
  /* Define the observed responses */
  float response[count] = {4.2, 2.2, 2.8, 5.0, 5.5};
  
  /* Define the Gaussian kernel with data
     dimension 2, standard deviation of 1,
     and a length scaling around .707 */
  KernType ker = kern(dim, 1, pow(2, -.5));
  
  /* Define the datapoints to extrapolate */
  float x_new[][dim] = {{.42, .33},
  					  {.95, .15}};
  
  pair<float, float> mv_1, mv_2;
  mv_1 = gpr_model(dim, count, &data[0][0], response, &x_new[0][0], ker);
  mv_2 = gpr_model(dim, count, &data[0][0], response, &x_new[1][0], ker);
	  cout << "(.42, .33) \\mapsto " << mv_1.first << " \\pm " << pow(mv_1.second, .5) << "( \\sigma^2 = " << mv_1.second << ")\\newline" << endl;
	cout << "(.95, .15) \\mapsto " << mv_2.first << " \\pm " << pow(mv_2.second, .5) << "( \\sigma^2 = " << mv_2.second << ")\\newline" << endl;
  }
