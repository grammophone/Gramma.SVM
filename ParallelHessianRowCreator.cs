using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Gramma.Kernels;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Gramma.SVM
{
	/// <summary>
	/// Computes a row of the Hessian using parallelism.
	/// Used by Hessian caches during cache misses.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public abstract class ParallelHessianRowCreator<T> : HessianRowCreator<T>
	{
		#region Private fields

		private int maxProcessorsCount;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <remarks>
		/// The <see cref="MaxProcessorsCount"/> to use
		/// defaults to the number of system processors.
		/// </remarks>
		public ParallelHessianRowCreator(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs, 
			Kernel<T> kernel)
			: base(trainingPairs, kernel)
		{
			this.maxProcessorsCount = Environment.ProcessorCount;
		}

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <param name="maxProcessorsCount">The maximum amount of processors to use, up to system available.</param>
		public ParallelHessianRowCreator(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs, 
			Kernel<T> kernel, 
			int maxProcessorsCount)
			: base(trainingPairs, kernel)
		{
			if (maxProcessorsCount > Environment.ProcessorCount) 
				throw new ArgumentException(
					"The requested processors count exceeds the available system processors", 
					"maxProcessorsCount");

			if (maxProcessorsCount <= 0)
				throw new ArgumentException(
					"The requested processors count must be positive",
					"maxProcessorsCount");

			this.maxProcessorsCount = maxProcessorsCount;
		}

		#endregion

		#region Public properties

		/// <summary>
		/// The maximum amount of processors to use.
		/// </summary>
		/// <remarks>
		/// If not explicitly specified by the appropriate constructor overload, 
		/// this defaults to the number of system processors.
		/// </remarks>
		public int MaxProcessorsCount
		{
			get
			{
				return this.maxProcessorsCount;
			}
			set
			{
				if (value > Environment.ProcessorCount)
					throw new ArgumentException(
						"The requested processors count exceeds the available system processors");

				if (value <= 0)
					throw new ArgumentException(
						"The requested processors count must be positive");

				this.maxProcessorsCount = value;
			}
		}

		#endregion

	}
}
