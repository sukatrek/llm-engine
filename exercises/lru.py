from doublell import DoublyLinkedList

class LRU:
    def __init__(self, capacity):
        self.map = dict()
        self.ll = DoublyLinkedList()
        self.capacity = capacity

    def get(self, key):
        if key not in self.map:
            return -1

        node = self.map[key]
        #we just accessed this value
        self.ll.move_to_front(node)
        return node.data


    def put(self, key, value):
        #insertion
        if key not in self.map:
            #we still have space for the node
            if len(self.map) < self.capacity:
                self.ll.insert_front(key, value)
                self.map[key] = self.ll.dummyHead.next
            #out of space
            else:
                #evict last
                node_to_remove = self.ll.dummyTail.prev
                key_to_remove = node_to_remove.key
                self.ll.remove(node_to_remove)

                del self.map[key_to_remove]
                #current stays the same since we are also adding in the node
                self.ll.insert_front(key, value)
                self.map[key] = self.ll.dummyHead.next
        #update the key
        else:
            #update the nodes data to be the new value
            self.map[key].data = value
            node = self.map[key]
            self.ll.move_to_front(node)




# Test 1: Basic get and put
lru = LRU(3)
lru.put(1, 10)
lru.put(2, 20)
lru.put(3, 30)

# lru.ll.print_ll()
# print(lru.map)

assert lru.get(1) == 10
assert lru.get(4) == -1  # doesn't exist

# Test 2: Eviction - least recently used gets evicted
lru = LRU(3)
lru.put(1, 10)
lru.put(2, 20)
lru.put(3, 30)
lru.get(1)       # 1 is now most recently used, order: 1, 3, 2


# lru.ll.print_ll()
# print(lru.map)

lru.put(4, 40)   # should evict 2 (LRU)


# lru.ll.print_ll()
# print(lru.map)

assert lru.get(2) == -1   # evicted
assert lru.get(1) == 10   # still there
assert lru.get(4) == 40   # still there

# Test 3: Update existing key
lru = LRU(2)
lru.put(1, 10)
lru.put(2, 20)
lru.put(1, 99)   # update key 1
assert lru.get(1) == 99
lru.put(3, 30)   # should evict 2 not 1
assert lru.get(2) == -1
assert lru.get(1) == 99

# Test 4: Capacity 1
lru = LRU(1)
lru.put(1, 10)
lru.put(2, 20)   # evicts 1
assert lru.get(1) == -1
assert lru.get(2) == 20

print("all tests passed")
