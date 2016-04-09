using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Vectors;
using Grammophone.Optimization;
using Grammophone.Kernels;

namespace Grammophone.SVM.CG
{
	public class CgChunkingNewtonBinaryClassifier<T>
		: CgChunkingBinaryClassifier<T, ConjugateGradient.TruncatedNewtonConstrainedMinimizeOptions>
	{
		#region Construction

		public CgChunkingNewtonBinaryClassifier(Kernel<T> kernel)
			: base(kernel)
		{
			//this.SolverOptions.StopCriterion =
			//  ε =>
			//    (Δw, H, Mg) => Δw.Norm2 / Δw.Length < ε || Mg.Norm2 / Mg.Length < ε;
		}

		#endregion

		#region CgChunkingBinaryClassifier<T> implementation

		protected override ConjugateGradient.SolutionCertificate Solve(
			ScalarFunction L, 
			VectorFunction dL, 
			TensorFunction d2L, 
			Vector λ0, 
			ScalarFunction φ, 
			VectorFunction dφ, 
			TensorFunction d2φ, 
			Func<double, VectorFunction> μ, 
			Func<Vector, bool> outOfDomainIndicator, 
			ConjugateGradient.ConstrainedMinimizePreconditioner M = null)
		{
			this.SolverOptions.DualityGap = (double)λ0.Length / 100000000.0;

			return ConjugateGradient.TruncatedNewtonConstrainedMinimize(
				dL,
				d2L,
				λ0,
				dφ,
				d2φ,
				μ,
				this.SolverOptions,
				outOfDomainIndicator,
				M);
		}

		#endregion
	}
}
