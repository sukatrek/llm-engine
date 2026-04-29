class Node:
    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    def __init__(self):
        self.dummyHead = Node(0, 0)
        self.dummyTail = Node(0, 0)
        self.dummyHead.next = self.dummyTail
        self.dummyTail.prev = self.dummyHead

    def insert_front(self, key, data):
        new_node = Node(key, data)
        new_node.next = self.dummyHead.next
        new_node.prev = self.dummyHead
        self.dummyHead.next.prev = new_node
        self.dummyHead.next = new_node


    def print_ll(self):
        curr = self.dummyHead
        while curr:
            print(curr.data, end=" ")
            curr = curr.next

    def remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev


    def move_to_front(self, node):
        #dh 5 3 2 1 dt
        node.next.prev = node.prev
        node.prev.next = node.next
        node.next = self.dummyHead.next
        node.prev = self.dummyHead
        self.dummyHead.next = node



if __name__ == "__main__":
    ll = DoublyLinkedList()

    ll.insert_front(1, 1)
    ll.insert_front(2, 2)
    ll.insert_front(3, 3)
    ll.insert_front(4, 4)
    ll.insert_front(5, 5)

    #this should remove 4
    ll.remove(ll.dummyHead.next.next)
    #move 2 to the front
    ll.move_to_front(ll.dummyHead.next.next.next)

    ll.print_ll()




