#!/usr/bin/env python3
# -*- coding: utf-8 -*-

强一致算法： Raft(基于log replicated的共识算法)
raft是更为简化的multi paxos（其实也就是上一个图中的paxos）算法，相比于paxos的复杂实现来说角色更少，问题更加精简。
rafti整体来说可以拆分成三个子问题来达成共识：
1、Leader Election 如何选择出leader
2、Log Replication 如何将log复制到其他的节点
3、Safety 保证复制之后集群的数据是一致的

重新定义了新的角色：
Leader 一个集群只有一个leader
Follower 一个服从leader决定的角色
Cadidate follower发现集群没有leader，会重新选举leader，参与选举的节点会变成candidate

###########################################################################################################################
第一个子问题：Leader Election的选举过程
初始所有节点都是follower状态，当开始选举的时候将待选举节node a点置为candidate状态
canditdate会向其他的follower节点发送请求进行leader投票，其他节点投票通过，则将candidate节点的状态置为leader状态。

接下来通过两种心跳时间来详细看一下选举过程的实现
在leader election选举的过程中会有两种timeout的设置来控制整个的选举过程：
    1、Election timeout
    表示follower等待请求的超时时间，如果这个时间结束前还没有收到来自leader或者选举leader的请求，那么当前follower便会变为 candidate。 raft给的设定值一般在150ms-300ms之间的随机值。
    变成candidate之后，当前节点会立即发送leader的投票请求，其他的follower节点会重置选举的超时时间。

    2、heartbeat timeout
    新选举出来的leader每隔一段时间会发送一个请求给follower，这个时间就是心跳时间。
    同样follower在相同的时间间隔内回复leader一个请求，表示自己已经收到。
    这样的心跳间隔将会一直持续，直到一个follower停止接受心跳，变成candidate。
    重新选举的过程就是candidate发送选举请求，follower接受到之后返回对应的心跳回应，candidate根据心跳回应的个数判断是否满足多数派，从而变成leader。变成leader之后，会持续发送心跳包来保证follower的存活。

###########################################################################################################################
第二个子问题：Log Replication过程
主要过程如下：
客户端发送请求到leader，leader接受之后将请求封装成log entry，并将log副本发送给其他的follower节点。
等待其他的follower节点回复，发现达到了多数派之后leader便将该entry写入到自己的文件系统之中；
写入之后再发送请求，表示follower节点也可以写入了。

接下来我们详细看看log Replicated的实现过程，我们的leader被选举出来之后用于请求的转发，将接受到的用户请求封装成log entry，并将log entry的副本转发给follower，这个log enry发送到follower之后也会用于重置follower的心跳时间。
1、客户端会发送一条请求到leader，这个请求会追加到leader的log上，但此时还没有写入leader所在节点的文件系统之上
2、这个条leader的log 会在leader发送下一条心跳包的时候携带该请求的信息 一起发送给follower
3、当entry提交到follower，且收到多数派的回复之后会给client一个回复，表示集群已经写入完成。同时将leader的log写入自己的文件系统，并且发送command让从节点也进行写入。这个过程就是multi paxos中的accepted阶段。

当存在网络分区情况时，raft也能保证数据的一致性。被分割出的非多数派集群将无法达到共识，即脑裂。
当集群再次连通时，将只听从最新任期Leader的指挥，旧Leader将退化为Follower，此时集群重新达到一致性状态。

不同语言实现的Raft, 具体详见：https://raft.github.io/

def main():
    pass


if __name__ == '__main__':
    main()
