using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Gramma.Kernels;

namespace Gramma.SVM
{
	/// <summary>
	/// Represents the class of a sample in a binary classification problem.
	/// </summary>
	[Serializable]
	public enum BinaryClass
	{
		/// <summary>
		/// The 'positive' class.
		/// </summary>
		Positive = 1,

		/// <summary>
		/// The 'negative' class.
		/// </summary>
		Negative = -1
	}

	/// <summary>
	/// Represents a binary classifier.
	/// </summary>
	/// <typeparam name="T">The type of items being classified.</typeparam>
	[Serializable]
	public abstract class BinaryClassifier<T>
	{
		#region Auxilliary types

		/// <summary>
		/// Specifies a sample item and its class.
		/// </summary>
		[Serializable]
		public struct TrainingPair
		{
			/// <summary>
			/// The sample item.
			/// </summary>
			public T Item;

			/// <summary>
			/// The class of the sample item.
			/// </summary>
			public BinaryClass Class;
		}

		#endregion

		#region Protected fields

		protected Kernel<T> kernel;

		#endregion

		#region Construction

		/// <summary>
		/// Create.
		/// </summary>
		/// <param name="kernel">
		/// The kernel used.
		/// </param>
		/// <remarks>
		/// After construction, use the <see cref="Train"/> method to perform learning.
		/// If the kernel already has components, the classifier will behave as already trained
		/// and will output predictions according to these components.
		/// </remarks>
		public BinaryClassifier(Kernel<T> kernel)
		{
			if (kernel == null) throw new ArgumentNullException("kernel");

			this.kernel = kernel + 1.0; // Compensate for the built-in w0.
		}

		#endregion

		#region Public properties

		/// <summary>
		/// Returns true if the classifier is trained.
		/// If false, invoking method <see cref="Discriminate"/> will return zero
		/// for any input.
		/// </summary>
		public bool IsTrained
		{
			get
			{
				return this.kernel.HasComponents;
			}
		}

		#endregion

		#region Public methods

		/// <summary>
		/// Discriminator estimator function over a given item.
		/// </summary>
		/// <param name="item">The item whose claass to estimate.</param>
		/// <returns>
		/// A positive value is returned when item is estimated 
		/// to be of the <see cref="BinaryClass.Positive"/> class, a negative when 
		/// of the <see cref="BinaryClass.Negative"/> class.
		/// </returns>
		public double Discriminate(T item)
		{
			return this.kernel.ComputeSum(item);
		}

		/// <summary>
		/// Train the classifier with the given training examples.
		/// Existing training, if any, will be cleared.
		/// </summary>
		/// <param name="trainingPairs">The training examples.</param>
		/// <param name="C">The slack (soft margin) variables penalty.</param>
		public void Train(IList<TrainingPair> trainingPairs, double C)
		{
			if (trainingPairs == null) throw new ArgumentNullException("trainingPairs");

			if (!trainingPairs.Any(p => p.Class == BinaryClass.Positive))
				throw new ArgumentException("There should be at least one positive example.", "trainingParis");

			if (!trainingPairs.Any(p => p.Class == BinaryClass.Negative))
				throw new ArgumentException("There should be at least one negative example.", "trainingParis");

			this.kernel.ClearComponents();

			this.TrainImplementation(trainingPairs, C);
		}

		#endregion

		#region Protected methods

		/// <summary>
		/// Override to implement training algorithm.
		/// </summary>
		/// <param name="trainingPairs">The training examples.</param>
		/// <param name="C">The slack (soft margin) variables penalty.</param>
		protected abstract void TrainImplementation(IList<TrainingPair> trainingPairs, double C);

		#endregion
	}
}
