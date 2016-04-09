using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Grammophone.Kernels;
using System.Threading.Tasks;

namespace Grammophone.SVM
{
	/// <summary>
	/// Computes a row of the Hessian using static parallelism.
	/// Appropriate when the execution time of a kernel is expected to be the same across
	/// different element pairs.
	/// Used by Hessian caches during cache misses.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	public class StaticParallelHessianRowCreator<T> : ParallelHessianRowCreator<T>
	{
		#region Private fields

		private ParallelQuery<Tuple<int, int>> trainingPairsPartitionerQuery;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <remarks>
		/// The <see cref="ParallelHessianRowCreator{T}.MaxProcessorsCount"/> to use
		/// defaults to the number of system processors.
		/// </remarks>
		public StaticParallelHessianRowCreator(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, Kernel<T> kernel)
			: base(trainingPairs, kernel)
		{
			initialize();
		}

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <param name="maxProcessorsCount">The maximum amount of processors to use, up to system available.</param>
		public StaticParallelHessianRowCreator(
			IList<BinaryClassifier<T>.TrainingPair> trainingPairs, 
			Kernel<T> kernel, 
			int maxProcessorsCount)
			: base(trainingPairs, kernel, maxProcessorsCount)
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

			//Parallel.ForEach(trainingPairsPartitioner, parallelOptions, (partitionRange, loopState) =>
			trainingPairsPartitionerQuery.ForAll(partitionRange =>
			{
				int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;

				for (int j = startIndex; j < endIndex; j++)
				{
					var trainingPair = trainingPairs[j];

					result[j] =
						(float)(yi * (double)trainingPair.Class * forkedKernel.ComputeSum(trainingPair.Item));
				}
			});

			return result;
		}

		#endregion

		#region Private methods

		private void initialize()
		{
			var trainingPairsPartitioner = new StaticRangePartitioner(0, trainingPairs.Count);

			this.trainingPairsPartitionerQuery =
				trainingPairsPartitioner
				.AsParallel()
				.AsUnordered()
				.WithExecutionMode(ParallelExecutionMode.ForceParallelism)
				.WithMergeOptions(ParallelMergeOptions.NotBuffered)
				.WithDegreeOfParallelism(this.MaxProcessorsCount);
		}

		#endregion
	}
}
