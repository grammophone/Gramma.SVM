using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Gramma.Kernels;

namespace Gramma.SVM
{
	/// <summary>
	/// Contract to provide an implementation of <see cref="HessianRowCreator{T}"/>
	/// when needed by a host who requires Hessian rows.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public interface IHessianRowCreatorProvider<T>
	{
		/// <summary>
		/// Supply a <see cref="HessianRowCreator{T}"/>.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <param name="maxProcessorsCount">The maximum amount of processors to use, up to system available.</param>
		/// <returns>Returns the requested <see cref="HessianRowCreator{T}"/>.</returns>
		HessianRowCreator<T> ProvideHessianRowCreator(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs,
			Kernel<T> kernel,
			int maxProcessorsCount);
	}
}
