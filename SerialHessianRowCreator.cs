using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Kernels;

namespace Grammophone.SVM
{
	/// <summary>
	/// Computes a row of the Hessian in a serial manner.
	/// Used by Hessian caches during cache misses.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public class SerialHessianRowCreator<T> : HessianRowCreator<T>
	{
		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		public SerialHessianRowCreator(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, Kernel<T> kernel)
			: base(trainingPairs, kernel)
		{
		}

		#endregion

		#region Public methods

		public override float[] ComputeRow(int rowIndex)
		{
			Kernel<T> forkedKernel = this.kernel.ForkNew();

			forkedKernel.AddComponent(1.0, this.trainingPairs[rowIndex].Item);

			double yi = (double)trainingPairs[rowIndex].Class;

			float[] result = new float[trainingPairs.Count];

			for (int j = 0; j < result.Length; j++)
			{
				var trainingPair = trainingPairs[j];

				result[j] =
					(float)(yi * (double)trainingPair.Class * forkedKernel.ComputeSum(trainingPair.Item));
			}

			return result;
		}

		#endregion
	}
}
