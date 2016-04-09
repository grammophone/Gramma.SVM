using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Grammophone.Vectors;
using Grammophone.Optimization;
using Grammophone.Kernels;

namespace Grammophone.SVM.CG
{
	public class CgChunkingLineSearchBinaryClassifier<T> : CgChunkingBinaryClassifier<T, ConjugateGradient.LineSearchConstrainedMinimizeOptions>
	{
		#region Construction

		public CgChunkingLineSearchBinaryClassifier(Kernel<T> kernel)
			: base(kernel)
		{
			this.SolverOptions.StopCriterion =
				ε =>
					(Δw, Mg) => Δw.Norm2 / Δw.Length < ε || Mg.Norm2 / Mg.Length < ε;
		}

		#endregion

		#region CgChunkingBinaryClassifier<T> implementation

		protected override ConjugateGradient.SolutionCertificate Solve(
			ScalarFunction L, 
			VectorFunction dL, 
			TensorFunction d2L, 
			Vector λ0, ScalarFunction φ, 
			VectorFunction dφ, 
			TensorFunction d2φ, Func<double, VectorFunction> μ, 
			Func<Vector, bool> outOfDomainIndicator, 
			ConjugateGradient.ConstrainedMinimizePreconditioner M = null)
		{
			return ConjugateGradient.LineSearchConstrainedMinimize(
				L,
				dL,
				λ0,
				φ,
				dφ,
				μ,
				this.SolverOptions,
				outOfDomainIndicator,
				M);
		}

		#endregion
	}
}
