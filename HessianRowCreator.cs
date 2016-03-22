using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Gramma.Kernels;

namespace Gramma.SVM
{
	/// <summary>
	/// Contract to create a row of the Hessian. 
	/// Used by Hessian caches during cache misses.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public abstract class HessianRowCreator<T>
	{
		#region Protected fields

		/// <summary>
		/// The kernel in use.
		/// </summary>
		protected Kernel<T> kernel;

		/// <summary>
		/// The training pairs.
		/// </summary>
		protected IList<BinaryClassifier<T>.TrainingPair> trainingPairs;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		public HessianRowCreator(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, Kernel<T> kernel)
		{
			if (kernel == null) throw new ArgumentNullException("kernel");
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");

			this.kernel = kernel;
			this.trainingPairs = trainingPairs;
		}

		#endregion

		#region Public methods

		/// <summary>
		/// Compute a row of the Hessian.
		/// </summary>
		/// <param name="rowIndex">The zero-based index of the Hessian row.</param>
		/// <returns>Returns the row as an float array of float values</returns>
		public abstract float[] ComputeRow(int rowIndex);

		#endregion
	}
}
