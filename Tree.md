

## Tree

```python3
class BinaryTree:
    def __init__(self,val):
        self.value = val
        self.leftChild = None
        self.rightChild = None
 
    def insertLeft(self,newNode):
        if self.leftChild == None:
            # add it as the left child if no left child , 如果没有左孩子，则设置为左孩子
            self.leftChild = BinaryTree(newNode)
        else:
            # add it as the left child, and change the previous one as the left child of the new left child.
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t
 
    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t
 
 
    def getRightChild(self):
        return self.rightChild
 
    def getLeftChild(self):
        return self.leftChild
 
    def setRootVal(self,val):
        self.value = val
 
    def getRootVal(self):
        return self.value
```


## Traversals

* **Inorder**: Left, Root, Right
* **Preorder**: Root, Left, Right
* **Postorder**: Left, Right, Root



```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root: return []
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```
