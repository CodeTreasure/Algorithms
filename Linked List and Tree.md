
## Linked List

regard every node as an object. the value is an attribute of the object, so is the next pointer. we use head to represent current sub linked list starting with head.

Let's look at the definition of a Linked List

```python3
# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
```



### Remove Duplicates from Sorted List (Leetcode #83)
https://leetcode.com/problems/remove-duplicates-from-sorted-list/

```python3
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 48ms
        temp = head
        while temp and temp.next:
            if temp.val == temp.next.val:
                temp.next = temp.next.next
            else:
                temp = temp.next
        return head
```
