# Machine Learning Toolbox

C++ library for basic machine learning usage. Support for perceptrons (classification and regression) and multilayered perceptron (classfication adn regression as well). A C# wrapper is also provided.

An unity project is provided, showing a graphical and interactiv representation of the perceptron classification and regression algorithm.

<p align="center">
  <img width="430" src="http://i.imgur.com/oqXhGD4.png"/>
  <img width="430" src="http://i.imgur.com/2IpsrPF.png"/>
</p>

## C++ Library
The directory MachineLearningToolbox contains the C++ library, and a test class with tests. The library is tested on Windows 10 and Mac OSX, and should work on Unix system as well.

## C# wrapper
The directory MachineLearningToolboxCSharp contains a C# wrapper for the C++ library, allowing it to use it more easily in any project supporting .Net DLL loading.

## Unity project
The directory MachineLearningUnity ontains an example of usage of the C++ library, with an interactive scene for testing the simple and multilayered perceptron.
