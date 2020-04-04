---
layout: post
title: 이진탐색(Binary Search)
subtitle: Data Structure, Algorithm
category: Data Structure, Algorithm
use_math: true
---

<br>
### 연결 리스트(Linked Lists)

단방향 연결 리스트(singly linked list)를 추상적 자료 구조로 정의해보자. 가할 수 있는 연산은 다음과 같다.

- 특정 원소 참조 (k 번째)

- 리스트 순회 (list traversal)

- 길이 얻어내기

- 원소의 삽입(insertion)

- 원소의 삭제(deletion)

- 원소의 합치기(concatenation)

<br>

이 중, 원소의 삽입(insertion)/삭제(deletion)/병합(concatenation)과 같은 연산이 쉽고 빠르게 이루어질 수 있다는 점이 Linked Lists가 Linear array에 비하여 가지는 장점이다.

즉, 이러한 연산을 필요로하는 경우, 연결 리스트를 사용하는 것 또한 유용하다.

```
class Node:

    def __init__(self, item):
        self.data = item
        self.next = None


class LinkedList:

    def __init__(self):
        self.nodeCount = 0
        self.head = None
        self.tail = None


    def __repr__(self):
        if self.nodeCount == 0:
            return 'LinkedList: empty'

        s = ''
        curr = self.head
        while curr is not None:
            s += repr(curr.data)
            if curr.next is not None:
                s += ' -> '
            curr = curr.next
        return s


    def getAt(self, pos):
        if pos < 1 or pos > self.nodeCount:
            return None

        i = 1
        curr = self.head
        while i < pos:
            curr = curr.next
            i += 1

        return curr


    def insertAt(self, pos, newNode):
        if pos < 1 or pos > self.nodeCount + 1:
            return False

        if pos == 1:
            newNode.next = self.head
            self.head = newNode

        else:
            if pos == self.nodeCount + 1:
                prev = self.tail
            else:
                prev = self.getAt(pos - 1)
            newNode.next = prev.next
            prev.next = newNode

        if pos == self.nodeCount + 1:
            self.tail = newNode

        self.nodeCount += 1
        return True


    def getLength(self):
        return self.nodeCount


    def traverse(self):
        result = []
        curr = self.head
        while curr is not None:
            result.append(curr.data)
            curr = curr.next
        return result


    def concat(self, L):
        self.tail.next = L.head
        if L.tail:
            self.tail = L.tail
        self.nodeCount += L.nodeCount

```
