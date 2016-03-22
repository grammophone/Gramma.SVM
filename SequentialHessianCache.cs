using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.Concurrent;
using System.Threading.Tasks;

using Gramma.Vectors;
using Gramma.Kernels;
using Gramma.Optimization;
using Gramma.Caching;

namespace Gramma.SVM
{
	/// <summary>
	/// Non-thread safe cache that stores a limited amount of rows of the Hessian of the goal of
	/// a full dual problem of an SVM.
	/// </summary>
	/// <typeparam name="T">The type of items operated by kernel.</typeparam>
	/// <remarks>
	/// The methods of this cache are not thread-safe.
	/// </remarks>
	public class SequentialHessianCache<T> : IHessianCache<T>
	{
		#region Auxilliary types

		/// <summary>
		/// Returns the two tensors involved in the decomposition of the dual problem
		/// by selecting an active set of Lagrange multipliers being optimized.
		/// </summary>
		/// <remarks>
		/// See 'Making Large-Scale SVM Learning Practical', Thorsten Joachims, 1998.
		/// </remarks>
		public class ActiveSubtensors
		{
			/// <summary>
			/// Create.
			/// </summary>
			/// <param name="QBB">
			/// Hessian submatrix consisting of rows and columns whose indices belong to 
			/// the working set.
			/// </param>
			/// <param name="QBN">
			/// Hessian submatrix consisting of rows whose indices belong to 
			/// the working set and columns belonging to the inactive set, or vice versa,
			/// since the Hessian is symmetric.
			/// </param>
			/// <param name="QBBd">
			/// The diagonal of the Hessian submatrix consisting of rows and columns 
			/// whose indices belonging to the working set.
			/// </param>
			/// <param name="Qa">
			/// Hessian submatrix consisting of rows whose indices belonging to 
			/// the working set.
			/// </param>
			internal ActiveSubtensors(Vector.Tensor QBB, Vector.Tensor QBN, Vector QBBd, Vector.Tensor Qa)
			{
				if (QBB == null) throw new ArgumentNullException("QBB");
				if (QBN == null) throw new ArgumentNullException("QBN");
				if (QBBd == null) throw new ArgumentNullException("QBBd");
				if (Qa == null) throw new ArgumentNullException("Qa");

				this.QBB = QBB;
				this.QBN = QBN;
				this.QBBd = QBBd;
				this.Qa = Qa;
			}

			/// <summary>
			/// Hessian submatrix consisting of rows and columns belonging to 
			/// the working set.
			/// </summary>
			public Vector.Tensor QBB { get; private set; }

			/// <summary>
			/// Hessian submatrix consisting of rows belonging to 
			/// the working set and columns belonging to the inactive set, or vice versa,
			/// since the Hessian is symmetric.
			/// </summary>
			/// <remarks>
			/// Since the Hessian is symetric, QBN = QNB.
			/// </remarks>
			public Vector.Tensor QBN { get; private set; }

			/// <summary>
			/// The diagonal of the Hessian submatrix consisting of rows and columns belonging to 
			/// the working set.
			/// </summary>
			public Vector QBBd { get; private set; }

			/// <summary>
			/// Hessian submatrix consisting of rows whose indices belonging to 
			/// the working set.
			/// </summary>
			public Vector.Tensor Qa { get; private set; }
		}

		#endregion

		#region Private fields

		private Kernel<T> kernel;

		private SequentialMRUCache<int, float[]> rowsCache;

		private IList<BinaryClassifier<T>.TrainingPair> trainingPairs;

		private float[] diagonal;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel.</param>
		/// <param name="maxCount">The maximum number of rows held in the cache.</param>
		public SequentialHessianCache(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs,
			Kernel<T> kernel,
			int maxCount = 1024)
		{
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");
			if (kernel == null) throw new ArgumentNullException("kernel");

			this.trainingPairs = trainingPairs;
			this.kernel = kernel;
			
			var rowCreator = new SerialHessianRowCreator<T>(trainingPairs, kernel);

			this.rowsCache = new SequentialMRUCache<int, float[]>(rowCreator.ComputeRow, maxCount);
		}

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel.</param>
		/// <param name="rowCreator">Creator of Hessian rows used in cache miss.</param>
		/// <param name="maxCount">The maximum number of rows held in the cache.</param>
		public SequentialHessianCache(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs,
			Kernel<T> kernel,
			HessianRowCreator<T> rowCreator,
			int maxCount = 1024)
		{
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");
			if (kernel == null) throw new ArgumentNullException("kernel");
			if (rowCreator == null) throw new ArgumentNullException("rowCreator");

			this.trainingPairs = trainingPairs;
			this.kernel = kernel;

			this.rowsCache = new SequentialMRUCache<int, float[]>(rowCreator.ComputeRow, maxCount);
		}

		#endregion

		#region Public properties

		/// <summary>
		/// The maximum number of rows held in the cache.
		/// </summary>
		public int MaxCount
		{
			get
			{
				return this.rowsCache.MaxCount;
			}
			set
			{
				this.rowsCache.MaxCount = value;
			}
		}

		/// <summary>
		/// The kernel in use.
		/// </summary>
		public Kernel<T> Kernel
		{
			get
			{
				return this.kernel;
			}
		}

		/// <summary>
		/// The training pairs of the problem.
		/// </summary>
		public IList<BinaryClassifier<T>.TrainingPair> TrainingPairs
		{
			get
			{
				return this.trainingPairs;
			}
		}

		#endregion

		#region Public methods

		/// <summary>
		/// Get the diagonal of the Hessian.
		/// </summary>
		public float[] GetDiagonal()
		{
			lock (this.rowsCache)
			{
				if (this.diagonal == null)
				{
					//var range = Enumerable.Range(0, this.trainingPairs.Count);

					//this.diagonal =
					//  range
					//  .AsParallel()
					//  .Select(i => (float)this.kernel.Compute(this.trainingPairs[i].Item, this.trainingPairs[i].Item))
					//  .ToArray();

					this.diagonal = new float[trainingPairs.Count];

					for (int i = 0; i < trainingPairs.Count; i++)
					{
						this.diagonal[i] = (float)this.kernel.Compute(this.trainingPairs[i].Item, this.trainingPairs[i].Item);
					}
				}
			}

			return diagonal;
		}

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
		public float[] GetRow(int rowIndex)
		{
			return this.rowsCache.Get(rowIndex);
		}

		/// <summary>
		/// Get statistics of cache usage.
		/// </summary>
		/// <returns>
		/// Returns a snapshot of cache statistics such as cumulative cache hits,
		/// total hits and cache items count.
		/// </returns>
		public SequentialMRUCache<int, float[]>.Statistics GetStatistics()
		{
			return this.rowsCache.GetStatistics();
		}

		/// <summary>
		/// Reset statistics concerning cache hits and total hits count.
		/// </summary>
		public void ResetStatistics()
		{
			this.rowsCache.ResetStatistics();
		}

		#endregion

		#region IDisposable Members

		/// <summary>
		/// Flushes away all elements from the cache.
		/// </summary>
		public void Dispose()
		{
			this.rowsCache.Clear();
		}

		#endregion

		#region Private methods

		private float[] ComputeRow(int rowindex)
		{
			Kernel<T> forkedKernel = this.kernel.ForkNew();

			forkedKernel.AddComponent(1.0, this.trainingPairs[rowindex].Item);

			double yi = (double)trainingPairs[rowindex].Class;

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
