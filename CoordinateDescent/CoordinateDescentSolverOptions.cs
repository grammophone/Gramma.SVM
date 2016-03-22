using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Gramma.SVM.CoordinateDescent
{
	/// <summary>
	/// Options for the algorithm used by <see cref="CoordinateDescentBinaryClassifier{T}"/>.
	/// </summary>
	[Serializable]
	public class CoordinateDescentSolverOptions
	{
		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		public CoordinateDescentSolverOptions()
		{
			this.CacheSize = 2048;
			this.ConstraintThreshold = 1e-5f;
			this.GradientThreshold = 2e-3f;
			this.ShrinkingPeriod = 1300;
			this.UseShrinking = true;
			this.MaxIterations = 400000;
		}

		#endregion

		#region Public properties

		/// <summary>
		/// Size of the cache containing Hessian rows. Default is 2048.
		/// </summary>
		public int CacheSize { get; set; }

		/// <summary>
		/// Threshold above which a solution of the dual constitutes a support vector.
		/// Default is 1e-5.
		/// </summary>
		public float ConstraintThreshold { get; set; }

		/// <summary>
		/// Threshold above which a normalized gradient of the dual problem is significant, making the
		/// corresponding dual variable eligible for update. When all normalized gradients
		/// fall below this value, the algorithm terminates. Default is 2e-3.
		/// </summary>
		public float GradientThreshold { get; set; }

		/// <summary>
		/// The period of iterations where shrinking is attempted, 
		/// if <see cref="UseShrinking"/> is enabled. Default is 1300.
		/// </summary>
		public int ShrinkingPeriod { get; set; }

		/// <summary>
		/// If set to true, the algorithm attempts every <see cref="ShrinkingPeriod"/>
		/// iterations to eliminate vectors which seem to be either non-support vectors or 
		/// bounded support vectors at the time. This reduces the amount of vectors being 
		/// manipulated considerably and speeds up subsequent iterations. But some of
		/// the eliminated vectors might prove to be actual support vectors, a false choice, 
		/// so the algorithm reenables all eliminated vectors before terminating.
		/// Default is true.
		/// </summary>
		public bool UseShrinking { get; set; }

		/// <summary>
		/// The maximum umber of iterations. Default is 400000.
		/// </summary>
		public int MaxIterations { get; set; }

		#endregion
	}
}
