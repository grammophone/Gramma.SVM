using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

using Grammophone.Linq;
using Grammophone.Vectors;
using Grammophone.Vectors.ExtraExtensions;
using Grammophone.Optimization;
using Grammophone.Kernels;

namespace Grammophone.SVM.CG
{
	public abstract class CgChunkingBinaryClassifier<T, SO> : ChunkingBinaryClassifier<T>
		where SO : ConjugateGradient.ConstrainedMinimizeOptions, new()
	{
		#region Private fields

		private SO solverOptions;

		#endregion

		#region Construction

		public CgChunkingBinaryClassifier(
			Kernel<T> kernel)
			: base(kernel)
		{
			this.solverOptions = new SO();
		}

		#endregion

		#region Public properties

		public SO SolverOptions
		{
			get
			{
				return this.solverOptions;
			}
			set
			{
				if (value == null) throw new ArgumentNullException("value");
				this.solverOptions = value;
			}
		}

		#endregion

		#region BinaryClassifier<T> implementation

		protected override void TrainImplementation(IList<BinaryClassifier<T>.TrainingPair> trainingPairs, double C)
		{
			//C = C / trainingPairs.Count;

			var hessianCache = new HessianCache<T>(trainingPairs, this.kernel, this.ChunkingOptions.CacheSize);

			// The gradient of the goal of the dual problem.
			Vector g = trainingPairs.Select(p => -1.0);

			// The dual problem's variables, which are the Lagrange multipliers.
			Vector α = trainingPairs.Select(p => 0.0);

			// Eliminated indices. Implements temporary shrinking.
			var eliminatedIndices = new HashSet<int>();

			var range = Enumerable.Range(0, trainingPairs.Count).ToArray();

			HashSet<int> previousActiveSet = new HashSet<int>();

			double gradientThreshold = this.ChunkingOptions.GradientThreshold;
			double constraintThreshold = this.ChunkingOptions.ConstraintThreshold;

			// Diagonal of the Hessian.
			Vector QD = hessianCache.GetDiagonal();

			while (true)
			{
				// Order indices by steepness of gradient, provided that the corresponding
				// points don't press the walls of constraints, 
				// nor are eliminated from previous iterations.
				var query = from i in range
										where !eliminatedIndices.Contains(i) && 
										(α[i] > constraintThreshold && α[i] < C - constraintThreshold && Math.Abs(g[i] / QD[i]) > gradientThreshold
										|| α[i] <= constraintThreshold && g[i] / QD[i] < -gradientThreshold
										|| α[i] >= C - constraintThreshold && g[i] / QD[i] > gradientThreshold)
										orderby Math.Abs(g[i] / QD[i])
										select i;

				var activeIndices = query.Take(this.ChunkingOptions.MaxChunkSize).ToArray();

				Trace.WriteLine(
					String.Format(
						"Chunking with active set of size {0} out of {1}.",
						activeIndices.Length,
						trainingPairs.Count));

				if (activeIndices.Length == 0 || activeIndices.All(index => previousActiveSet.Contains(index)))
				{
					if (eliminatedIndices.Count == 0)
					{
						break; // We reached our goal. Break out of the loop.
					}
					else
					{
						Trace.WriteLine(
							String.Format(
								"No more eligible variables. Clearing {0} eliminated variables and restarting.",
								eliminatedIndices.Count));

						eliminatedIndices.Clear();

						continue;
					}
				}

				//if (activeIndices.All(index => previousActiveSet.Contains(index))) break;

				previousActiveSet.Clear();

				foreach (var index in activeIndices)
				{
					previousActiveSet.Add(index);
				}

				var inactiveIndices = hessianCache.GetInactiveIndices(activeIndices);

				// Choose variables in working set.
				Vector αw = from index in activeIndices
										select α[index];

				// Choose variables in inactive set.
				Vector αi = from index in inactiveIndices
										select α[index];

				var activeSubtensors = hessianCache.GetActiveSubtensors(activeIndices, inactiveIndices);

				Trace.WriteLine(hessianCache.GetStatistics());
				hessianCache.ResetStatistics();

				var QBB = activeSubtensors.QBB;
				var QBN = activeSubtensors.QBN;
				var QBBd = activeSubtensors.QBBd;
				var Qa = activeSubtensors.Qa;

				// Vector of size of the active set consisting of all 1.0's.
				Vector ew = activeIndices.Select(i => 1.0);

				// Setup of chunked problem.

				var activeRange = Enumerable.Range(0, activeIndices.Length).ToArray();

				Vector gc = QBN(αi) - ew;

				ScalarFunction L = // Lagrangian.
					λ =>
						0.5 * λ * QBB(λ) + gc * λ;

				VectorFunction dL = // Lagrangian gradient.
					λ =>
						QBB(λ) + gc;

				TensorFunction d2L = // Hessian of the Lagrangian.
					λ => QBB;

				VectorFunction d2Ld = // Diagonal of the Hessian of the Lagrangian.
					λ => QBBd;

				ScalarFunction φ = // Log barrier function of the constraints.
					λ =>
						-λ.AsParallel().Sum(λi => Math.Log(λi) + Math.Log(C - λi));

				VectorFunction dφ = // Gradient of the log barrier.
					λ =>
						-λ.AsParallel().Select(λi => 1.0 / λi + 1.0 / (λi - C));

				TensorFunction d2φ = // Hessian of the log barrier.
					λ =>
						y =>
							activeRange.AsParallel().Select(i => (1.0 / λ[i].Squared() + 1.0 / (λ[i] - C).Squared()) * y[i]);

				VectorFunction d2φd = // Diagonal of the Hessian of the log barrier.
					λ =>
						activeRange.AsParallel().Select(i => 1.0 / λ[i].Squared() + 1.0 / (λ[i] - C).Squared());

				var constraintsRange = Enumerable.Range(0, activeIndices.Length * 2).ToArray();

				Func<double, VectorFunction> μ = // Lagrange multiplier estimator: Dual of the dual.
					t =>
						λ =>
							constraintsRange.Select(i =>
								i < activeIndices.Length ?
								1.0 / (t * λ[i]) :
								1.0 / (t * (C - λ[i - activeIndices.Length]))
							);

				Func<Vector, bool> outOfDomainIndicator = // Returns true when λ is out of domain.
					λ =>
						λ.Any(λi => λi < 0 || λi > C);

				ConjugateGradient.ConstrainedMinimizePreconditioner M = // Jacobi preconditioner.
					t =>
						λ =>
							Vector.GetDiagonalTensor(
								(t * d2Ld(λ) + d2φd(λ)).Select(Hii => 1.0 / Hii)
							);

				Vector αw0 = activeIndices.Select(i => C / 2.0);

				var certificate = this.Solve( // Do the trick.
					L,
					dL,
					d2L,
					αw0,
					φ,
					dφ,
					d2φ,
					μ,
					outOfDomainIndicator,
					M);

				// Update gradient of the whole dual problem.
				Vector αwold = αw;

				αw = certificate.Optimum;

				Vector Δαw = αw - αwold;

				// If no progress is made, eliminate the active indices 
				// for the next iteration, shrinking the problem.
				//if (Δαw.Norm2 / Δαw.Length < this.ChunkingOptions.ConstraintThreshold / 20.0)
				//if (Δαw.All(Δαwi => Δαwi.Squared() < this.ChunkingOptions.ConstraintThreshold))
				//{
				//  Trace.WriteLine(
				//    String.Format(
				//      "Shrinking by {0} items, eliminating {1} in total.",
				//      activeIndices.Length,
				//      activeIndices.Length + eliminatedIndices.Count));

				//  foreach (var index in activeIndices)
				//  {
				//    eliminatedIndices.Add(index);
				//  }
				//}
				//else if (eliminatedIndices.Count > 0)
				//{
				//  // Since a change is made to the gradient, 
				//  // call off all eliminations from gradient search.
				//  eliminatedIndices.Clear();

				//  Trace.WriteLine("End of shrinking. All variables return to play.");
				//}

				// Update the gradient.
				g += Qa(Δαw);

				// Update solution.
				for (int i = 0; i < activeIndices.Length; i++)
				{
					α[activeIndices[i]] = αw[i];
				}

				//var eliminationQuery = from i in activeRange
				//                       where !(
				//                       αw[i] > this.ChunkingOptions.ConstraintThreshold && αw[i] < C - this.ChunkingOptions.ConstraintThreshold ||
				//                       αw[i] < this.ChunkingOptions.ConstraintThreshold && g[i] < this.ChunkingOptions.GradientThreshold |
				//                       αw[i] > C - this.ChunkingOptions.ConstraintThreshold && g[i] > -this.ChunkingOptions.GradientThreshold)
				//                       select activeIndices[i];

				//var newEliminatedIndices = eliminationQuery.ToArray();

				//if (newEliminatedIndices.Length > 128)
				//{
				//  Trace.WriteLine(
				//    String.Format(
				//      "Shrinking by {0} items, eliminating {1} in total.",
				//      newEliminatedIndices.Length,
				//      newEliminatedIndices.Length + eliminatedIndices.Count));

				//  foreach (var index in newEliminatedIndices)
				//  {
				//    eliminatedIndices.Add(index);
				//  }
				//}
			}

			Trace.WriteLine("Training finished.");

			// Add the support vectors as kernel components.

			var supportVectorTuples = from i in range
																where α[i] > this.ChunkingOptions.ConstraintThreshold
																select new
																{
																	Weight = α[i] * (double)trainingPairs[i].Class,
																	SupportVector = trainingPairs[i].Item
																};

			Trace.WriteLine(String.Format("Support vectors found: {0}.", supportVectorTuples.Count()));

			foreach (var supportVectorTuple in supportVectorTuples)
			{
				this.kernel.AddComponent(supportVectorTuple.Weight, supportVectorTuple.SupportVector);
			}

			//for (int i = 0; i < α.Length; i++)
			//{
			//  if (α[i] > this.ChunkingOptions.ConstraintThreshold)
			//  {
			//    double yi = (double)trainingPairs[i].Class;
			//    this.kernel.AddComponent(α[i] * yi, trainingPairs[i].Item);
			//  }
			//}

		}

		#endregion

		#region Protected methods

		protected abstract ConjugateGradient.SolutionCertificate Solve(
			ScalarFunction L,
			VectorFunction dL,
			TensorFunction d2L,
			Vector λ0,
			ScalarFunction φ,
			VectorFunction dφ,
			TensorFunction d2φ,
			Func<double, VectorFunction> μ,
			Func<Vector, bool> outOfDomainIndicator,
			ConjugateGradient.ConstrainedMinimizePreconditioner M = null);

		#endregion
	}
}
