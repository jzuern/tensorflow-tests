
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>


int main(){

  std::cout << " --------- Sum reduction ----------- \n";

  Eigen::Tensor<float, 5, Eigen::ColMajor> T_in(2, 2, 3, 2, 3);
  T_in.setValues({{{{{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{5.0f, 3.0f, 4.0f},
                 {6.0f, 0.0f, 2.0f}}},
               {{{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{5.0f, 3.0f, 4.0f},
                 {6.0f, 0.0f, 2.0f}}}},
                 {{{{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{5.0f, 3.0f, 4.0f},
                 {6.0f, 0.0f, 2.0f}}},
               {{{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{0.0f, 1.0f, 2.0f},
                 {8.0f, 9.0f, 1.0f}},
                {{5.0f, 3.0f, 4.0f},
                 {6.0f, 0.0f, 2.0f}}}}});


  Eigen::array<int, 3> dims({2,3,4}); // reducing along dim 2,3, and 4
  Eigen::Tensor<float, 2, Eigen::ColMajor> T_out = T_in.sum(dims);

  std::cout << "T_out:\n" << std::endl << T_out << std::endl << std::endl;



}
