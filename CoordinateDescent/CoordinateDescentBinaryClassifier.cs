using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Grammophone.Kernels;

namespace Grammophone.SVM.CoordinateDescent
{
	/// <summary>
	/// A binary classifier using a modified SMO-like algorithm.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	/// <remarks>
	/// Instead of updating two coordinates at a time, as the standard SMO algorithm does in order
	/// to deal with the equality constraint which arises in the dual problem of the SVM,
	/// this algorithm lifts the equality constraint and thus updates one coordinate at a time,
	/// like a standard coordinate descent algorithm, resulting in a substancial simplification.
	/// </remarks>
	[Serializable]
	public abstract class CoordinateDescentBinaryClassifier<T> : BinaryClassifier<T>
	{
		#region Private fields

		private CoordinateDescentSolverOptions solverOptions;

		#endregion

		#region Construction

		public CoordinateDescentBinaryClassifier(Kernel<T> kernel)
			: base(kernel)
		{
			this.solverOptions = new CoordinateDescentSolverOptions();
		}

		#endregion

		#region Public properties

		/// <summary>
		/// Options for the algorithm.
		/// </summary>
		public CoordinateDescentSolverOptions SolverOptions
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
	}
}
