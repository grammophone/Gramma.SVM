using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Kernels;
using Grammophone.Linq;
using Grammophone.Optimization;
using Grammophone.Vectors;
using Grammophone.Vectors.ExtraExtensions;
using System.Collections.Concurrent;
using System.Threading.Tasks;

namespace Grammophone.SVM.CG
{
	/// <summary>
	/// Binary classifier trained using conjugate gradent method.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	/// <typeparam name="O">
	/// The type of solver options, 
	/// a descendant of <see cref="ConjugateGradient.ConstrainedMinimizeOptions"/>.
	/// </typeparam>
	[Serializable]
	public abstract class CgBinaryClassifier<T, O> : BinaryClassifier<T>
		where O : ConjugateGradient.ConstrainedMinimizeOptions
	{
		#region Auxilliary types

		private struct SingedGram
		{
			public Single[,] Matrix;
			public Vector Diagonal;
		}

		#endregion

		#region Private fields

		private O solverOptions;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		public CgBinaryClassifier(Kernel<T> kernel, O solverOptions)
			: base(kernel)
		{
			if (solverOptions == null) throw new ArgumentNullException("solverOptions");

			this.solverOptions = solverOptions;
		}

		#endregion

		#region Public properties

		public O SolverOptions
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
			int P = trainingPairs.Count;

			//C = C / P; // Normalize slack penalty.

			int constraintCount = 2 * P; // The count of constraints implied by 0 <= λi <= C.

			var range = Enumerable.Range(0, P).ToArray(); // Range for enumerating samples.

			Func<int, T> x = // Training samples.
				i => trainingPairs[i].Item;

			Func<int, double> d = // Class indicators of training samples.
				i => (double)trainingPairs[i].Class;

			var signedGram = this.GetSignedGramMatrix(trainingPairs);

			float[,] Q = signedGram.Matrix;

			Vector.Tensor H = Vector.GetTensor(Q);

			Vector Qd = signedGram.Diagonal;

			Vector g = range.Select(i => -1.0);

			ScalarFunction L = // Lagrangian.
				λ => 0.5 * range.AsParallel().Sum(i => range.Sum(j => λ[i] * λ[j] * Q[i, j]))
					- λ.Sum();
			// 0.5 * λ * H(λ) * λ -λ.Sum();

			VectorFunction dL = // Lagrangian gradient.
				λ =>
					range.AsParallel().Select(i => range.Sum(j => Q[i, j] * λ[j]) - 1.0);
			//H(λ) + g;

			TensorFunction d2L = // Hessian of the Lagrangian.
				λ => H;

			VectorFunction d2Ld = // Diagonal of the Hessian of the Lagrangian.
				λ => Qd;

			ScalarFunction φ = // Log barrier function of the constraints.
				λ =>
					-λ.AsParallel().Sum(λi => Math.Log(λi) + Math.Log(C - λi));

			VectorFunction dφ = // Gradient of the log barrier.
				λ =>
					-λ.AsParallel().Select(λi => 1.0 / λi + 1.0 / (λi - C));

			TensorFunction d2φ = // Hessian of the log barrier.
				λ =>
					y =>
						range.AsParallel().Select(i => (1.0 / λ[i].Squared() + 1.0 / (λ[i] - C).Squared()) * y[i]);

			VectorFunction d2φd = // Diagonal of the Hessian of the log barrier.
				λ =>
					range.AsParallel().Select(i => 1.0 / λ[i].Squared() + 1.0 / (λ[i] - C).Squared());

			Func<double, VectorFunction> μ = // Lagrange multiplier estimator: Dual of the dual.
				t =>
					λ =>
						range.Select(i =>
							i < P ?
							1.0 / (t * λ[i]) :
							1.0 / (t * (C - λ[i - P]))
						);

			Func<Vector, bool> outOfDomainIndicator = // Returns true when λ is out of domain.
				λ =>
					λ.Any(λi => λi < 0 || λi > C);

			Func<int, ScalarFunction> fc = // The constraint functions: 0 <= λ[i] <=  C
				i =>
					λ =>
						i < P ?
						-λ[i] :
						λ[i - P] - C;

			Func<int, VectorFunction> dfc = // Gradients of the constraint functions.
				i =>
					λ =>
						i < P ?
						range.Select(j => (i == j) ? -1.0 : 0.0) :
						range.Select(j => (i - P == j) ? 1.0 : 0.0);

			Func<int, TensorFunction> d2fc = // Hessians of the contraint functions.
				i =>
					λ => Vector.ZeroTensor;

			Vector zero = new Vector(P);

			Func<int, VectorFunction> d2fcd = // Diagonals of Hessians of the constraint functions.
				i =>
					λ => zero;

			Vector λ0 = range.Select(i => C / 2.0);

			ConjugateGradient.ConstrainedMinimizePreconditioner M = // Jacobi preconditioner.
				t =>
					λ =>
						Vector.GetDiagonalTensor(
							(t * d2Ld(λ) + d2φd(λ)).Select(Hii => 1.0 / Hii)
						);

			// Do the trick.
			var certificate = this.Solve(
				L,
				dL,
				d2L,
				λ0,
				φ,
				dφ,
				d2φ,
				μ,
				outOfDomainIndicator,
				M);

			for (int i = 0; i < P; i++)
			{
				var λi = certificate.Optimum[i];

				if (λi > this.solverOptions.DualityGap)
				{
					this.kernel.AddComponent(d(i) * λi, x(i));
				}
			}

		}

		private SingedGram GetSignedGramMatrix(IList<BinaryClassifier<T>.TrainingPair> trainingPairs)
		{
			var partitioner = Partitioner.Create(trainingPairs, true);

			var Q = new float[trainingPairs.Count, trainingPairs.Count];

			var Qd = new Vector(trainingPairs.Count);

			ParallelOptions options = new ParallelOptions();

			Parallel.For(0, trainingPairs.Count, i =>
			{
				var ti = trainingPairs[i];

				int di = (int)ti.Class;

				T xi = ti.Item;

				for (int j = i; j < trainingPairs.Count; j++)
				{
					var tj = trainingPairs[j];

					int dj = (int)tj.Class;

					T xj = tj.Item;

					double Qij = di * dj * this.kernel.Compute(xi, xj);

					Q[i, j] = (float)Qij;

					if (i == j)
					{
						Qd[i] = Qij;
					}
					else
					{
						Q[j, i] = (float)Qij;
					}
				}
			}
			);

			return new SingedGram { Matrix = Q, Diagonal = Qd };
		}

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
