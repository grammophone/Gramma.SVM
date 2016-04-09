using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using Grammophone.Vectors;
using Grammophone.Kernels;
using Grammophone.Optimization;

namespace Grammophone.SVM.CoordinateDescent
{
	/// <summary>
	/// A binary classifier using a modified SMO-like algorithm running serially.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	/// <remarks>
	/// Instead of updating two coordinates at a time, as the standard SMO algorithm does in order
	/// to deal with the equality constraint which arises in the dual problem of the SVM,
	/// this algorithm lifts the equality constraint and thus updates one coordinate at a time,
	/// like a standard coordinate descent algorithm, resulting in a substancial simplification.
	/// </remarks>
	[Serializable]
	public class SerialCoordinateDescentBinaryClassifier<T> : CoordinateDescentBinaryClassifier<T>
	{
		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="kernel">The kernel to use for items of type <typeparamref name="T"/>.</param>
		public SerialCoordinateDescentBinaryClassifier(Kernel<T> kernel)
			: base(kernel)
		{
		}

		#endregion

		#region Protected methods

		protected override void TrainImplementation(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, double _C)
		{
			float C = (float)_C;

			var stopWatch = new Stopwatch();

			stopWatch.Start();

			bool useShrinking = this.SolverOptions.UseShrinking;

			using (var hessianCache = new SequentialHessianCache<T>(
				trainingPairs,
				kernel,
				new SerialHessianRowCreator<T>(trainingPairs, kernel),
				this.SolverOptions.CacheSize))
			{

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

				unsafe
				{
					fixed (float* pα = α)
					fixed (float* pQD = QD, pg = g, pgs = gs)
					fixed (int* pWorkingIndices = workingIndices)
					fixed (int* pNewWorkingIndices = newWorkingIndices)
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

							int activeIndex = -1;

							float ΔGmax = -1.0f;

							{
								int* pCurrentWorkingIndex = pWorkingIndices;

								for (int k = 0; k < workingIndicesCount; k++, pCurrentWorkingIndex++)
								{
									int i = *pCurrentWorkingIndex;

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

										if (ΔG > ΔGmax)
										{
											activeIndex = i;
											ΔGmax = ΔG;
										}
									}


								}

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

										pCurrentWorkingIndex = pWorkingIndices;

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
							}

							float[] QW = hessianCache.GetRow(activeIndex);

							fixed (float* pQW = QW)
							{
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

										float* pgsk = pgs;
										float* pQWk = pQW;

										for (int k = 0; k < trainingPairsCount; k++, pgsk++, pQWk++)
										{
											*pgsk -= C * (*pQWk);
										}
									}
								}
								else
								{
									αB = C;

									if (useShrinking && αBold < C)
									{
										//var Δgs = C * QW;
										//gs += Δgs;

										float* pgsk = pgs;
										float* pQWk = pQW;

										for (int k = 0; k < trainingPairsCount; k++, pgsk++, pQWk++)
										{
											*pgsk += C * (*pQWk);
										}
									}
								}

								// Update gradient.
								//g += (αB - α[activeIndex]) * QW;

								var ΔαB = αB - αBold;

								int* pCurrentWorkingIndex = pWorkingIndices;

								for (int k = 0; k < workingIndicesCount; k++, pCurrentWorkingIndex++)
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

								// Update solution.
								pα[activeIndex] = αB;

								// Should we shrink?
								if (useShrinking && iterationsCount % shrinkingPeriod == shrinkingPeriodRemainder)
								{
									// Yes.
									int newWorkingIndicesCount = 0;

									pCurrentWorkingIndex = pWorkingIndices;

									int* pCurrentNewWorkingIndex = pNewWorkingIndices;

									for (int k = 0; k < workingIndicesCount; k++, pCurrentWorkingIndex++)
									{
										int i = *pCurrentWorkingIndex;

										float αi = pα[i];
										var gi = pg[i];

										// New working points: These ones not pressing hard against the constraints.
										if (αi > 0.0 && αi < C
											|| αi == 0.0 && gi < 0 // gradientThreshold
											|| αi == C && gi > 0 //-gradientThreshold
											)
										{
											*(pCurrentNewWorkingIndex++) = i;
										}
									}

									newWorkingIndicesCount = (int)(pCurrentNewWorkingIndex - pNewWorkingIndices);

									if (newWorkingIndicesCount < workingIndicesCount - 12)
									{
										Trace.WriteLine(
											String.Format(
												"Shrinking by {0}.",
												workingIndicesCount - newWorkingIndicesCount));

										Array.Copy(newWorkingIndices, workingIndices, newWorkingIndicesCount);
										workingIndicesCount = newWorkingIndicesCount;
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
		}

		#endregion
	}
}
