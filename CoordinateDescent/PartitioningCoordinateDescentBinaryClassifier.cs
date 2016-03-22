using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using Gramma.Vectors;
using Gramma.Kernels;
using Gramma.Optimization;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Threading;

namespace Gramma.SVM.CoordinateDescent
{
	/// <summary>
	/// A binary classifier using a modified SMO-like algorithm running using parallelization of loops.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	/// <remarks>
	/// Instead of updating two coordinates at a time, as the standard SMO algorithm does in order
	/// to deal with the equality constraint which arises in the dual problem of the SVM,
	/// this algorithm lifts the equality constraint and thus updates one coordinate at a time,
	/// like a standard coordinate descent algorithm, resulting in a substancial simplification.
	/// </remarks>
	[Serializable]
	public class PartitioningCoordinateDescentBinaryClassifier<T> : CoordinateDescentBinaryClassifier<T>
	{
		#region Delegate types

		/// <summary>
		/// Delegate to supply a <see cref="HessianRowCreator{T}"/>.
		/// </summary>
		/// <param name="trainingPairs">The training pairs.</param>
		/// <param name="kernel">The kernel in use.</param>
		/// <param name="maxProcessorsCount">The maximum amount of processors to use, up to system available.</param>
		/// <returns>Returns the requested <see cref="HessianRowCreator{T}"/>.</returns>
		public delegate HessianRowCreator<T> HessianRowCreatorProviderDelegate(
			IList<TrainingPair> trainingPairs, 
			Kernel<T> kernel, 
			int maxProcessorsCount);

		#endregion

		#region Private fields

		private int maxProcessorsCount;

		private HessianRowCreatorProviderDelegate hessianRowCreatorProvider;

		private volatile float ΔGmax;

		private volatile int volatileActiveIndex;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="kernel">The kernel to use for items of type <typeparamref name="T"/>.</param>
		public PartitioningCoordinateDescentBinaryClassifier(Kernel<T> kernel)
			: base(kernel)
		{
			this.maxProcessorsCount = Environment.ProcessorCount;
			
			this.hessianRowCreatorProvider = 
				(trainingPairs, kern, maxProcessorsCount) => new LoadBalancingParallelHessianRowCreator<T>(trainingPairs, kern, maxProcessorsCount);
		}

		#endregion

		#region Public properties

		/// <summary>
		/// The maximum processors count to be used. 
		/// This is an upper bound depending on the thread pool.
		/// Default is the number of the processors available.
		/// </summary>
		public int MaxProcessorsCount
		{
			get
			{
				return this.maxProcessorsCount;
			}
			set
			{
				if (value < 1) 
					throw new ArgumentException("Value must be positive", "value");
				
				if (value > Environment.ProcessorCount) 
					throw new ArgumentException(String.Format("Value exceeds the {0} available processors", Environment.ProcessorCount));

				this.maxProcessorsCount = value;
			}
		}

		/// <summary>
		/// Provides an implementation of <see cref="HessianRowCreator{T}"/>
		/// when needed by a host who requires Hessian rows.
		/// Default setting is a provider 
		/// which supplies a <see cref="LoadBalancingParallelHessianRowCreator{T}"/>.
		/// </summary>
		public HessianRowCreatorProviderDelegate HessianRowCreatorProvider
		{
			get
			{
				return this.hessianRowCreatorProvider;
			}
			set
			{
				if (value == null) throw new ArgumentNullException("value");

				this.hessianRowCreatorProvider = value;
			}
		}

		#endregion

		#region Protected methods

		protected override void TrainImplementation(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, double _C)
		{
			float C = (float)_C;

			var stopWatch = new Stopwatch();

			stopWatch.Start();

			bool useShrinking = this.SolverOptions.UseShrinking;

			var hessianRowCreator = 
				this.HessianRowCreatorProvider(trainingPairs, kernel, maxProcessorsCount);

			var hessianCache = new SequentialHessianCache<T>(
				trainingPairs,
				kernel,
				hessianRowCreator,
				this.SolverOptions.CacheSize);

			// The gradient of the goal of the dual problem.
			float[] g = trainingPairs.Select(p => -1.0f).ToArray();

			// The dual problem's variables, which are the Lagrange multipliers.
			float[] α = trainingPairs.Select(p => 0.0f).ToArray();

			var range = Enumerable.Range(0, trainingPairs.Count).ToArray();

			int trainingPairsCount = trainingPairs.Count;

			int maxIterations = this.SolverOptions.MaxIterations;

			int iterationsCount;
			int shrinkingPeriod = this.SolverOptions.ShrinkingPeriod;
			int shrinkingPeriodRemainder = this.SolverOptions.ShrinkingPeriod - 1;

			float gradientThreshold = (float)this.SolverOptions.GradientThreshold;

			var initialIndices = Enumerable.Range(0, trainingPairs.Count).ToList();

			var workingIndices = initialIndices.ToArray();
			int workingIndicesCount = trainingPairs.Count;

			var newWorkingIndices = new int[trainingPairs.Count];

			var gs = new float[trainingPairs.Count];

			// The diagonal of the Hessian.
			float[] QD = hessianCache.GetDiagonal();

			shrinkingPeriod = 2;

			Partitioner<Tuple<int, int>> trainingIndicesPartitioner = //Partitioner.Create(0, trainingPairsCount);
				new StaticRangePartitioner(0, trainingPairsCount);

			var trainingIndicesQuery = 
				trainingIndicesPartitioner
				.AsParallel()
				.WithDegreeOfParallelism(maxProcessorsCount);

			Partitioner<Tuple<int, int>> workingIndicesPartitioner = trainingIndicesPartitioner;

			var workingIndicesQuery = trainingIndicesQuery;

			SpinLock activeIndexLock = new SpinLock();

			unsafe
			{
				fixed (float* _pα = α)
				fixed (float* _pQD = QD, _pg = g, _pgs = gs)
				fixed (int* _pWorkingIndices = workingIndices)
				fixed (int* _pNewWorkingIndices = newWorkingIndices)
				{
					float* pα = _pα;
					float* pg = _pg;
					float* pgs = _pgs;
					float* pQD = _pQD;

					int* pWorkingIndices = _pWorkingIndices;
					int* pNewWorkingIndices = _pNewWorkingIndices;

					for (iterationsCount = 0; iterationsCount < maxIterations; iterationsCount++)
					{

						if (iterationsCount % 10000 == 0)
							Trace.WriteLine(String.Format("Iteration #{0}.", iterationsCount));

						shrinkingPeriod = Math.Min(shrinkingPeriod + 4, this.SolverOptions.ShrinkingPeriod);
						shrinkingPeriodRemainder = shrinkingPeriod - 1;

						//var steepestSearch = from i in range
						//                     where //!eliminatedIndices.Contains(i) &&
						//                     α[i] > 0.0 && α[i] < C && Math.Abs(g[i]) > gradientThreshold
						//                     || α[i] <= 0.0 && g[i] < -gradientThreshold
						//                     || α[i] >= C && g[i] > gradientThreshold
						//                     select i;

						//var workingIndices = steepestSearch.ToArray();

						// Find the most violating coordinate in the active set of coordinates.

						volatileActiveIndex = -1;

						ΔGmax = -1.0f;

						workingIndicesQuery.ForAll(partitionRange =>
						{
							int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;

							int* pCurrentWorkingIndex = &pWorkingIndices[startIndex];

							float ΔGmaxLocal = -1.0f;
							int activeIndexLocal = -1;

							for (int k = startIndex; k < endIndex; k++, pCurrentWorkingIndex++)
							{
								int i = workingIndices[k];

								var αi = pα[i];

								var gi = pg[i];

								// This is the negative of the unconstrained 1-dimensional newton step.
								float normalized_gi = gi / pQD[i];

								//// What would be the unconstrained update?
								//var αB = αi - gi;

								//// Trim the update to the constraints.
								//if (αB > C)
								//{
								//  gi = αi - C;
								//  αB = C;
								//}
								//else if (αB < 0.0)
								//{
								//  gi = αi;
								//  αB = 0.0;
								//}

								//var ΔG = 0.5 * QDi * (αi * αi - αB * αB) + (gi - QDi * αi) * (αi - αB);

								if (αi < C && normalized_gi < -gradientThreshold
									|| αi > 0.0 && normalized_gi > gradientThreshold)
								{
									float ΔG = gi * normalized_gi;

									if (ΔG > ΔGmaxLocal)
									{
										activeIndexLocal = i;
										ΔGmaxLocal = ΔG;
									}
								}

							}

							bool activeIndexLockAcquired = false;

							activeIndexLock.Enter(ref activeIndexLockAcquired);

							if (activeIndexLockAcquired)
							{
								if (ΔGmaxLocal > ΔGmax)
								{
									volatileActiveIndex = activeIndexLocal;
									ΔGmax = ΔGmaxLocal;
								}

								activeIndexLock.Exit(true);
							}

						});

						int activeIndex = volatileActiveIndex;

						// Did we find a significant slope?
						if (activeIndex == -1)
						{
							// No. So, do we have a shrinked active set?
							if (workingIndicesCount == trainingPairsCount)
							{
								// No. This means we are done.
								break;
							}
							else
							{
								// Else, unshrink and try again.

								Trace.WriteLine(
									String.Format(
										"Unshrinking {0} items.",
										trainingPairs.Count - workingIndicesCount));

								// Reconstruct the gradient.

								//g = range.Select(i => -1.0);

								float* pgk = pg;

								for (int k = 0; k < trainingPairsCount; k++, pgk++)
								{
									*pgk = -1.0f;
								}

								int *pCurrentWorkingIndex = pWorkingIndices;

								for (int j = 0; j < workingIndicesCount; j++)
								{
									int i = *(pCurrentWorkingIndex++);

									var αi = pα[i];

									if (αi > 0.0 && αi < C)
									{
										var Q = hessianCache.GetRow(i);

										fixed (float* pQ = Q)
										{
											//g += αi * Q;

											pgk = pg;

											float* pQk = pQ;

											for (int k = 0; k < trainingPairsCount; k++, pgk++, pQk++)
											{
												*pgk += αi * (*pQk);
											}
										}
									}
								}

								//g += gs;

								pgk = pg;
								float* pgsk = pgs;

								for (int k = 0; k < trainingPairsCount; k++, pgk++, pgsk++)
								{
									*pgk += *pgsk;
								}

								initialIndices.CopyTo(workingIndices);
								workingIndicesCount = initialIndices.Count;

								shrinkingPeriod = 2;

								Trace.WriteLine("Unshrinked.");

								continue;
							}
						}

						float[] QW = hessianCache.GetRow(activeIndex);

						fixed (float* _pQW = QW)
						{
							float* pQW = _pQW;

							float QBB = pQW[activeIndex];

							var αBold = pα[activeIndex];

							float αB = αBold - pg[activeIndex] / QBB;

							// Clip the solution: 0 <= αB <= C
							if (αB < C)
							{
								if (αB < 0.0)
								{
									αB = 0.0f;
								}

								if (useShrinking && αBold == C)
								{
									//var Δgs = C * QW;
									//gs -= Δgs;

									trainingIndicesQuery.ForAll(partitionRange =>
									{
										int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;

										float* pgsk = &pgs[startIndex];
										float* pQWk = &pQW[startIndex];

										for (int k = startIndex; k < endIndex; k++, pgsk++, pQWk++)
										{
											*pgsk -= C * (*pQWk);
										}
									});
								}
							}
							else
							{
								αB = C;

								if (useShrinking && αBold < C)
								{
									//var Δgs = C * QW;
									//gs += Δgs;

									trainingIndicesQuery.ForAll(partitionRange =>
									{
										int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;

										float* pgsk = &pgs[startIndex];
										float* pQWk = &pQW[startIndex];

										for (int k = startIndex; k < endIndex; k++, pgsk++, pQWk++)
										{
											*pgsk += C * (*pQWk);
										}
									});
								}
							}

							// Update gradient.
							//g += (αB - α[activeIndex]) * QW;

							var ΔαB = αB - αBold;

							workingIndicesQuery.ForAll(partitionRange =>
							{
								int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;

								int* pCurrentWorkingIndex = &pWorkingIndices[startIndex];

								for (int k = startIndex; k < endIndex; k++, pCurrentWorkingIndex++)
								{
									int i = *pCurrentWorkingIndex;

									// Is this the active index, and the optimum point is not clipped?
									//if (i == activeIndex && αB > 0.0 && αB < C)
									//{
									//  // Then there's no need to compute ΔαB * QW[i], the gradient should be zero.
									//  pg[i] = 0.0f;
									//}
									//else
									{
										// ...else, update gradient.
										pg[i] += ΔαB * QW[i];
									}
								}
							});

							// Update solution.
							pα[activeIndex] = αB;

							// Should we shrink?
							if (useShrinking && iterationsCount % shrinkingPeriod == shrinkingPeriodRemainder)
							{
								// Yes.

								int newWorkingIndicesCount = -1;

								int* pCurrentNewWorkingIndex = pNewWorkingIndices;

								//workingIndicesQuery.ForAll(partitionRange =>
								{

									//int startIndex = partitionRange.Item1, endIndex = partitionRange.Item2;
									int startIndex = 0, endIndex = workingIndicesCount;

									int* pCurrentWorkingIndex = &pWorkingIndices[startIndex];

									for (int k = startIndex; k < endIndex; k++, pCurrentWorkingIndex++)
									{
										int i = *pCurrentWorkingIndex;

										float αi = pα[i];
										var gi = pg[i] /* / pQD[i] */;

										// New working points: These ones not pressing hard against the constraints.
										if (αi > 0.0 && αi < C
											|| αi == 0.0 && gi < 0.0f // gradientThreshold
											|| αi == C && gi > 0.0f //-gradientThreshold
											)
										{
											//pNewWorkingIndices[Interlocked.Increment(ref newWorkingIndicesCount)] = i;
											//*(pCurrentNewWorkingIndex++) = i;

											int newWorkingIndexOffset = Interlocked.Increment(ref newWorkingIndicesCount);

											*(pNewWorkingIndices + newWorkingIndexOffset) = i;
										}
									}

								}
								//);

								newWorkingIndicesCount++;

								//newWorkingIndicesCount = (int)(pCurrentNewWorkingIndex - pNewWorkingIndices);

								if (newWorkingIndicesCount < workingIndicesCount - 12)
								{
									Trace.WriteLine(
										String.Format(
											"Shrinking by {0}.",
											workingIndicesCount - newWorkingIndicesCount));

									//Array.Copy(newWorkingIndices, workingIndices, newWorkingIndicesCount);
									for (int i = 0; i < newWorkingIndicesCount; i++)
									{
										pWorkingIndices[i] = pNewWorkingIndices[i];
									}

									workingIndicesCount = newWorkingIndicesCount;

									workingIndicesPartitioner = //Partitioner.Create(0, workingIndicesCount);
										new StaticRangePartitioner(0, workingIndicesCount);

									workingIndicesQuery =
										workingIndicesPartitioner
										.AsParallel()
										.WithDegreeOfParallelism(maxProcessorsCount)
										;
								}
							}
						}
					}
				}
			}

			// Add the support vectors as kernel components.

			var supportVectorTuples = from i in range
																where α[i] > this.SolverOptions.ConstraintThreshold
																select new
																{
																	Weight = α[i] * (double)trainingPairs[i].Class,
																	SupportVector = trainingPairs[i].Item
																};

			stopWatch.Stop();

			Trace.WriteLine(String.Format("Finished after {0} iterations.", iterationsCount));
			Trace.WriteLine(String.Format("Time elapsed: {0} sec.", stopWatch.Elapsed.TotalSeconds));
			Trace.WriteLine(
				String.Format(
					"Training samples: {0}, support vectors: {1}.",
					trainingPairs.Count,
					supportVectorTuples.Count()));

			Trace.WriteLine(hessianCache.GetStatistics());

			foreach (var supportVectorTuple in supportVectorTuples)
			{
				this.kernel.AddComponent(supportVectorTuple.Weight, supportVectorTuple.SupportVector);
			}
		}

		#endregion
	}
}
