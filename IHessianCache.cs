using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Kernels;

namespace Grammophone.SVM
{
	/// <summary>
	/// Basic Hessian cache behavior.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public interface IHessianCache<T> : IDisposable
	{
		#region Properties

		/// <summary>
		/// The kernel in use.
		/// </summary>
		Kernel<T> Kernel
		{
			get;
		}

		/// <summary>
		/// The training pairs of the problem.
		/// </summary>
		IList<BinaryClassifier<T>.TrainingPair> TrainingPairs
		{
			get;
		}

		#endregion

		#region Methods

		/// <summary>
		/// Get the diagonal of the Hessian.
		/// </summary>
		float[] GetDiagonal();

		/// <summary>
		/// Get a row from the cache or recompute one on cache miss.
		/// </summary>
		/// <param name="rowIndex">The zero-based row index.</param>
		/// <returns>
		/// Returns a vector representing the row.
		/// </returns>
		/// <remarks>
		/// If there is a cache miss, the newly computed row is inserted as the most recently
		/// used row in the cache.
		/// </remarks>
		float[] GetRow(int rowIndex);

		#endregion
	}
}
