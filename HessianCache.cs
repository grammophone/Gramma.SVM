using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.Concurrent;
using System.Threading.Tasks;

using Grammophone.Vectors;
using Grammophone.Vectors.ExtraExtensions;
using Grammophone.Kernels;
using Grammophone.Optimization;
using Grammophone.Caching;

namespace Grammophone.SVM
{
	/// <summary>
	/// Cache that stores a limited amount of rows of the Hessian of the goal of
	/// a full dual problem of an SVM.
	/// </summary>
	/// <typeparam name="T">The type of items operated by kernel.</typeparam>
	/// <remarks>
	/// The methods of this cache are thread-safe, suitable for parallel algorithms.
	/// </remarks>
	public class HessianCache<T>
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

		private MRUCache<int, float[]> rowsCache;

		private IList<BinaryClassifier<T>.TrainingPair> trainingPairs;

		private Vector diagonal;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel.</param>
		/// <param name="maxCount">The maximum number of rows held in the cache.</param>
		public HessianCache(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs, 
			Kernel<T> kernel, 
			int maxCount = 1024)
		{
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");
			if (kernel == null) throw new ArgumentNullException("kernel");

			this.trainingPairs = trainingPairs;
			this.kernel = kernel;
			this.rowsCache = new MRUCache<int, float[]>(this.ComputeRow, maxCount);
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

		#endregion

		#region Public methods

		/// <summary>
		/// Get the diagonal of the Hessian.
		/// </summary>
		public Vector GetDiagonal()
		{
			lock (this.rowsCache)
			{
				if (this.diagonal == null)
				{
					var range = Enumerable.Range(0, this.trainingPairs.Count);

					this.diagonal = 
						range
						.AsParallel()
						.Select(i => this.kernel.Compute(this.trainingPairs[i].Item, this.trainingPairs[i].Item));
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

		///// <summary>
		///// Get a tensor representing a submatrix of the Hessian consisting on a subset of its
		///// rows.
		///// </summary>
		///// <param name="rowIndices">The indices of the rows.</param>
		///// <returns>
		///// Returns a Tensor for the submatrix.
		///// </returns>
		///// <remarks>
		///// The returned Tensor does not retrieve the rows when this method is called, but 
		///// the rows are retrieved from the cache as needed when a vector is applied to it.
		///// </remarks>
		//public Vector.Tensor GetRowsSubTensor(IList<int> rowIndices)
		//{
		//  if (rowIndices == null) throw new ArgumentNullException("rowIndices");

		//  return
		//    y =>
		//      rowIndices.AsParallel().Select(i => GetRow(i) * y);
		//}

		/// <summary>
		/// Get the tensors involved in the decomposition of the full Hessian to a subset of 
		/// active indices.
		/// </summary>
		/// <param name="activeIndices">The indices specifying the active set.</param>
		/// <returns>Returns the two tensors of the decomposition.</returns>
		public ActiveSubtensors GetActiveSubtensors(IList<int> activeIndices)
		{
			if (activeIndices == null) throw new ArgumentNullException("activeIndices");

			// Find the complementary set of the active set, the fixed set.
			IList<int> complementaryIndices = GetInactiveIndices(activeIndices);

			return GetActiveSubtensors(activeIndices, complementaryIndices);
		}

		/// <summary>
		/// Given the working set, get the complementary set, the fixed set.
		/// </summary>
		/// <param name="activeIndices">The indices specifying the active (working) set.</param>
		/// <returns>Returns the indices of the inactive (fixed) set.</returns>
		public int[] GetInactiveIndices(IList<int> activeIndices)
		{
			if (activeIndices == null) throw new ArgumentNullException("activeIndices");

			bool[] indicesMap = new bool[this.trainingPairs.Count];

			for (int k = 0; k < activeIndices.Count; k++)
			{
				indicesMap[activeIndices[k]] = true;
			}

			List<int> complementaryIndices = new List<int>(this.trainingPairs.Count);

			for (int k = 0; k < indicesMap.Length; k++)
			{
				if (!indicesMap[k]) complementaryIndices.Add(k);
			}

			return complementaryIndices.ToArray();
		}

		/// <summary>
		/// Get the tensors involved in the decomposition of the full Hessian to a subset of 
		/// active indices.
		/// </summary>
		/// <param name="activeIndices">The indices specifying the working set.</param>
		/// <param name="inactiveIndices">The indices specifying the inactive set.</param>
		/// <returns>Returns the two tensors of the decomposition.</returns>
		public ActiveSubtensors GetActiveSubtensors(
			IList<int> activeIndices, 
			IList<int> inactiveIndices)
		{
			if (activeIndices == null) throw new ArgumentNullException("activeIndices");
			if (inactiveIndices == null) throw new ArgumentNullException("inactiveIndices");

			// Retrieve the rows coresponding to the active set.
			float[][] rows = new float[activeIndices.Count][];

			Vector activeDiagonal = new Vector(activeIndices.Count);

			var activeRange = Enumerable.Range(0, activeIndices.Count);

			var partitioner = Partitioner.Create(activeRange.ToArray(), true);

			//this.rowsCache.Clear(); // TODO: For debug only! Remove!

			//partitioner.AsParallel().ForAll(
			//  i =>
			//  {
			//    Vector row = this.GetRow(activeIndices[i]);
			//    activeDiagonal[i] = row[activeIndices[i]];
			//    rows[i] = row;
			//  });

			Parallel.ForEach(
				partitioner, 
				// new ParallelOptions { MaxDegreeOfParallelism = 1 },
				i =>
				{
					float[] row = this.GetRow(activeIndices[i]);
					activeDiagonal[i] = row[activeIndices[i]];
					rows[i] = row;
				});

			Vector.Tensor QBB =
				y =>
					activeRange.AsParallel()
					.Select(i => activeRange.Sum(j => rows[i][activeIndices[j]] * y[j]));

			var inactiveRange = Enumerable.Range(0, inactiveIndices.Count);

			Vector.Tensor QBN =
				y =>
					activeRange.AsParallel()
					.Select(i => inactiveRange.Sum(j => rows[i][inactiveIndices[j]] * y[j]));

			var fullRange = Enumerable.Range(0, activeIndices.Count + inactiveIndices.Count);

			Vector.Tensor Qa =
			  y =>
					fullRange.AsParallel()
					.Select(i => activeRange.Sum(j => rows[j][i] * y[j]));

			return new ActiveSubtensors(QBB, QBN, activeDiagonal, Qa);
		}

		/// <summary>
		/// Get statistics of cache usage.
		/// </summary>
		/// <returns>
		/// Returns a snapshot of cache statistics such as cumulative cache hits,
		/// total hits and cache items count.
		/// </returns>
		public MRUCache<int, float[]>.Statistics GetStatistics()
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
