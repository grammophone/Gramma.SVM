using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.Concurrent;

namespace Grammophone.SVM
{
	public class StaticPartitioner<T> : Partitioner<T>
	{
		#region Auxilliary classes

		internal class StaticPartitionerEnumerator : IEnumerator<T>
		{
			#region Private fields

			private IList<T> masterList;

			int partitionStart, partitionEnd, currentPosition;

			#endregion

			#region Construction

			public StaticPartitionerEnumerator(IList<T> masterList, int partitionStart, int partitionEnd)
			{
				this.masterList = masterList;
				this.partitionStart = partitionStart;
				this.partitionEnd = partitionEnd;
				this.currentPosition = partitionStart - 1;
			}

			#endregion

			#region IEnumerator<T> Members

			public T Current
			{
				get
				{
					return this.masterList[currentPosition];
				}
			}

			#endregion

			#region IDisposable Members

			public void Dispose()
			{
			}

			#endregion

			#region IEnumerator Members

			object System.Collections.IEnumerator.Current
			{
				get
				{
					return masterList[currentPosition];
				}
			}

			public bool MoveNext()
			{
				return ++currentPosition < partitionEnd;
			}

			public void Reset()
			{
				currentPosition = partitionStart - 1;
			}

			#endregion
		}

		#endregion

		#region Private fields

		private IList<T> items;

		#endregion

		#region Construction

		public StaticPartitioner(IList<T> items)
		{
			if (items == null) throw new ArgumentNullException("items");

			this.items = items;
		}

		#endregion

		#region Public methods

		public override IList<IEnumerator<T>> GetPartitions(int partitionCount)
		{
			List<IEnumerator<T>> partitions = new List<IEnumerator<T>>(partitionCount);

			int partitionBaseSize = items.Count / partitionCount;

			int partitionRemainder = items.Count % partitionCount;

			int partitionStart = 0;

			for (int i = 0; i < partitionCount; i++)
			{
				int partitionEnd = partitionStart + partitionBaseSize;

				if (partitionRemainder > 0)
				{
					partitionEnd++;
					partitionRemainder--;
				}

				partitions.Add(new StaticPartitionerEnumerator(items, partitionStart, partitionEnd));

				partitionStart = partitionEnd;
			}

			return partitions;
		}

		#endregion
	}
}
