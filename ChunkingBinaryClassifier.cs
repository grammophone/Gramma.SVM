using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Grammophone.Kernels;

namespace Grammophone.SVM
{
	public abstract class ChunkingBinaryClassifier<T> : BinaryClassifier<T>
	{
		#region Auxilliary types

		public class Options
		{
			#region Private fields

			private int maxChunkSize;

			private double constraintThreshold;

			private double gradientThreshold;

			private int cacheSize;

			#endregion

			#region Construction

			public Options()
			{
				this.maxChunkSize = 1000;
				this.constraintThreshold = 1e-3;
				this.gradientThreshold = 1e-3;
				this.cacheSize = 2048;
			}

			#endregion

			#region Public properties

			/// <summary>
			/// The maximum sizeof a chunk. Default is 1000.
			/// </summary>
			public int MaxChunkSize
			{
				get
				{
					return this.maxChunkSize;
				}
				set
				{
					if (value <= 0)
						throw new ArgumentException("MaxChunkSize must be positive");

					this.maxChunkSize = value;
				}
			}

			/// <summary>
			/// The least distance a variable is pushed against a constraint, below which
			/// te variable is no longer considered for further optimization. Default is 1e-3.
			/// </summary>
			public double ConstraintThreshold
			{
				get
				{
					return this.constraintThreshold;
				}
				set
				{
					if (value <= 0.0)
						throw new ArgumentException("ConstraintThreshold must be positive");

					this.constraintThreshold = value;
				}
			}

			/// <summary>
			/// The gradient value below which a variable is not considered for
			/// further optimization. Default is 1e-3.
			/// </summary>
			public double GradientThreshold
			{
				get
				{
					return this.gradientThreshold;
				}
				set
				{
					if (value <= 0.0) throw new ArgumentException("GradientThreshold must be positive");
					this.gradientThreshold = value;
				}
			}

			/// <summary>
			/// The maximum number of Hessian rows retained in the cache. Default is 2048.
			/// </summary>
			public int CacheSize
			{
				get
				{
					return this.cacheSize;
				}
				set
				{
					this.cacheSize = value;
				}
			}

			#endregion
		}

		#endregion

		#region Private fields

		private Options chunkingOptions;

		#endregion

		#region Construction

		public ChunkingBinaryClassifier(
			Kernel<T> kernel)
			: base(kernel)
		{
			this.chunkingOptions = new Options();
		}

		#endregion

		#region Public properties

		public Options ChunkingOptions
		{
			get
			{
				return chunkingOptions;
			}
			set
			{
				if (value == null) throw new ArgumentNullException("value");
				this.chunkingOptions = value;
			}
		}

		#endregion

	}
}
