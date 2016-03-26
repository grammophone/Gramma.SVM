# Gramma.SVM
This .NET library implements Support vector Machines. It works with kernels of any type using the [Gramma.Kernels](https://github.com/grammophone/Gramma.Kernels) library.

Binary classifier implementations are derived from the abstract `BinaryClassifier<T>`, where `T` is the type of items being classified. The implementations need a `Kernel<T>` to be supplied upon creation, they are trained via the `Train` method and they discriminate via the `Discriminate` method which returns Double instead of Boolean. This permits the output to be refitted to other values, for example to [map it to probabilities using Platt's method (1999)](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639). The trained SVM can be saved and loaded using standard .NET serialization.

In the UML diagram below, we can see the main SVM implementations of the library. These implementations follow the ['Coordinate Descent' method](http://dx.doi.org/10.1109/CIES.2013.6611732) for serial and parallel computations. The serial implementation is `SerialCoordinateDescentBinaryClassifier<T>`, the parallel one is `PartitioningCoordinateDescentBinaryClassifier<T>`. Both of them derive from `CoordinateDescentBinaryClassifier<T>` from which they inherit a `SolverOptions` property of type `CoordinateDescentSolverOptions`.

![SVM hierarchy](http://s7.postimg.org/5m8a04z6z/SVM_hierarchy.png)

There also exist other implementations not seen in this diagram, but these are experimental and not intended for use.

This project relies on the following projects being in sibling directories:
* [Gramma.Caching](https://github.com/grammophone/Gramma.Caching)
* [Gramma.Indexing](https://github.com/grammophone/Gramma.Indexing)
* [Gramma.Kernels](https://github.com/grammophone/Gramma.Kernels)
* [Gramma.Linq](https://github.com/grammophone/Gramma.Linq)
* [Gramma.Optimization](https://github.com/grammophone/Gramma.Optimization)
* [Gramma.Vectors](https://github.com/grammophone/Gramma.Vectors)
