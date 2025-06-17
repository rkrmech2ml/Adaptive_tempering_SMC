from sklearn.datasets import load_iris

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    def _insert(self, root, key):
        if key < root.val:
            if root.left is None:
                root.left = Node(key)
            else:
                self._insert(root.left, key)
        else:
            if root.right is None:
                root.right = Node(key)
            else:
                self._insert(root.right, key)

    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.val, end=' ')
            self.inorder(root.right)

# Example usage with open source data (Iris dataset sepal lengths)

data = load_iris()
sepal_lengths = data.data[:, 0]

bt = BinaryTree()
for val in sepal_lengths:
    bt.insert(val)

print("Inorder traversal of the binary tree:")
bt.inorder(bt.root)
print()