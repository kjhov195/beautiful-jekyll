---
layout: post
title: 양방향 연결 리스트(Doubly Linked Lists)
subtitle: Data Structure, Algorithm
category: Data Structure, Algorithm
use_math: true
---

<br>
### 양방향 연결 리스트(Doubly Linked Lists)

앞서 살펴본 단방향 연결 리스트(singly linked list)는 링크가 한 방향으로 연결되어 있다. 즉, 앞의 노드로부터 그 뒤의 노드를 향하는 방향으로만 연결되어 있는 것이다.

반면 양방향 연결 리스트(doubly linked list)는 인접한 노드들이 뒤쪽 방향 뿐만아니라, 앞쪽으로도 연결되어 있다. 즉, 특정 한 노드를 살펴보면, 앞의 노드와 연결되는 링크와 뒤로 이어지는 노드를 연결하는 링크를 둘 다 가지고 있는 것이다. 이 때문에 이러한 구조의 연결 리스트를 Doubly Linked Lists라고 부른다.


```
class Node:

    def __init__(self, item):
        self.data = item
        self.prev = None
        self.next = None


class DoublyLinkedList:

    def __init__(self):
        self.nodeCount = 0
        self.head = Node(None)
        self.tail = Node(None)
        self.head.prev = None
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = None


    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'

        s = ''
        curr = self.head
        while curr.next.next:
            curr = curr.next
            s += repr(curr.data)
            if curr.next.next is not None:
                s += ' -> '
        return s


    def getLength(self):
        return self.nodeCount


    def traverse(self):
        result = []
        curr = self.head
        while curr.next.next:
            curr = curr.next
            result.append(curr.data)
        return result


    def getAt(self, pos):
        if pos < 0 or pos > self.nodeCount:
            return None

        if pos > self.nodeCount // 2:
            i = 0
            curr = self.tail
            while i < self.nodeCount - pos + 1:
                curr = curr.prev
                i += 1
        else:
            i = 0
            curr = self.head
            while i < pos:
                curr = curr.next
                i += 1

        return curr


    def insertAfter(self, prev, newNode):
        next = prev.next
        newNode.prev = prev
        newNode.next = next
        prev.next = newNode
        next.prev = newNode
        self.nodeCount += 1
        return True


    def insertAt(self, pos, newNode):
        if pos < 1 or pos > self.nodeCount + 1:
            return False

        prev = self.getAt(pos - 1)
        return self.insertAfter(prev, newNode)
```
