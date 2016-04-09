using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections.Concurrent;

namespace Grammophone.SVM
{
	public class StaticRangePartitioner : OrderablePartitioner<Tuple<int, int>>
	{
		#region Auxilliary classes

		internal class SinglePartitionEnumerator<P> : IEnumerator<P>
		{
			#region Private fields

			private P partition;

			private bool canMoveNext;

			#endregion

			#region Construction

			public SinglePartitionEnumerator(P partition)
			{
				this.partition = partition;
				this.canMoveNext = true;
			}

			#endregion

			#region IEnumerator<Tuple<int,int>> Members

			public P Current
			{
				get { return partition; }
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
				get { return partition; }
			}

			public bool MoveNext()
			{
				bool answer = canMoveNext;

				canMoveNext = false;

				return answer;
			}

			public void Reset()
			{
				canMoveNext = true;
			}

			#endregion
		}

		#endregion

		#region Private fields

		private int rangeStart, rangeEnd;

		#endregion

		#region Construction

		public StaticRangePartitioner(int rangeStart, int rangeEnd)
			: base(true, true, true)
		{
			if (rangeStart > rangeEnd)
				throw new ArgumentException("rangeStart should not be greater than rangeEnd");

			this.rangeStart = rangeStart;
			this.rangeEnd = rangeEnd;
		}

		#endregion

		#region Public methods

		public override IList<IEnumerator<Tuple<int, int>>> GetPartitions(int partitionCount)
		{
			var partitions = new List<IEnumerator<Tuple<int, int>>>(partitionCount);

			int rangeSize = rangeEnd - rangeStart;

			int partitionBaseSize = rangeSize / partitionCount;

			int partitionRemainder = rangeSize % partitionCount;

			int partitionStart = rangeStart;

			for (int i = 0; i < partitionCount; i++)
			{
				int partitionEnd = partitionStart + partitionBaseSize;

				if (partitionRemainder > 0)
				{
					partitionEnd++;
					partitionRemainder--;
				}

				partitions.Add(new SinglePartitionEnumerator<Tuple<int, int>>(new Tuple<int, int>(partitionStart, partitionEnd)));

				partitionStart = partitionEnd;
			}

			return partitions;

		}

		public override IList<IEnumerator<KeyValuePair<long, Tuple<int, int>>>> GetOrderablePartitions(int partitionCount)
		{
			var partitions = new List<IEnumerator<KeyValuePair<long, Tuple<int, int>>>>(partitionCount);

			int rangeSize = rangeEnd - rangeStart;

			int partitionBaseSize = rangeSize / partitionCount;

			int partitionRemainder = rangeSize % partitionCount;

			int partitionStart = rangeStart;

			for (int i = 0; i < partitionCount; i++)
			{
				int partitionEnd = partitionStart + partitionBaseSize;

				if (partitionRemainder > 0)
				{
					partitionEnd++;
					partitionRemainder--;
				}

				var tuple = new Tuple<int, int>(partitionStart, partitionEnd);

				partitions.Add(new SinglePartitionEnumerator<KeyValuePair<long, Tuple<int, int>>>(new KeyValuePair<long, Tuple<int, int>>(i, tuple)));

				partitionStart = partitionEnd;
			}

			return partitions;
		}

		#endregion

	}
}
