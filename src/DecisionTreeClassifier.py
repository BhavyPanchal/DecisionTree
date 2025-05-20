import pandas as pd
import math

class DecisionTreeNode:
	"""
	Nodes of the decision tree
	"""
	def __init__(self, value=None, cNode=True, attr=None):
		self.value = value
		self.__cNode = cNode #cNode stand for condition nodes aka non-leaf nodes
		self.attr = attr
		self.left: DecisionTreeNode = None
		self.right: DecisionTreeNode = None

	def isCNode(self):
		return self.__cNode

	def __repr__(self):
		return f"DecisionTreeNode(value: {self.value}, isCNode: {self.__cNode})"

class DecisionTreeClassifier:
	"""
	desicion tree base class
	"""
	def __init__(self) -> None:
		self.root = DecisionTreeNode()

	def train(self, df: pd.DataFrame) -> None:
		"""
		Train the model from a pandas DataFrame object
		meant to be trained only once

		labels must be set as index of the dataframe
		"""
		baseEntropy = self.entropy(df)
		self.__bestFit(baseEntropy, df)

	def __bestFit(self, entropy, data: pd.DataFrame, root=None):
		if not root:
			root = self.root

		maxIg = 0
		bestFit = None
		bestFitColumnName = None
		bestFitLeftData = None
		bestFitRightData = None

		for columnName, column in data.items():
			for value in column:
				left = data[data[columnName] <= value]
				right = data[data[columnName] > value]

				ig = self.informationGain(entropy, left, right)
				if ig > maxIg:
					maxIg = ig
					bestFit = value
					bestFitColumnName = columnName
					bestFitRightData = right
					bestFitLeftData = left

		self.__addBestFitToTree(root, bestFit, bestFitColumnName, bestFitLeftData, bestFitRightData)

	def __addBestFitToTree(self, root: DecisionTreeNode, bestFit, attr, left: pd.DataFrame, right: pd.DataFrame):
		root.value = bestFit
		root.attr = attr
		leftEntropy = self.entropy(left)
		rightEntropy = self.entropy(right)

		if leftEntropy == 0:
			root.left = DecisionTreeNode(left.index[0], False, attr=attr)
		else:
			root.left = DecisionTreeNode()
			self.__bestFit(leftEntropy, left, root.left)
		
		if rightEntropy == 0:
			root.right = DecisionTreeNode(right.index[0], False, attr=attr)
		else:
			root.right = DecisionTreeNode()
			self.__bestFit(rightEntropy, right, root.right)

	@staticmethod
	def informationGain(parentEntropy, left: pd.DataFrame, right: pd.DataFrame):
		n1 = left.__len__()
		n2 = right.__len__()

		return parentEntropy - (((n1 / (n1 + n2)) * DecisionTreeClassifier.entropy(left)) + (n2 / (n1 + n2) * DecisionTreeClassifier.entropy(right)))

	@staticmethod
	def entropy(df: pd.DataFrame) -> float:
		classes = {}
		for Class in df.index:
			if Class == None:
				continue
			classes.update({Class: classes.get(Class, 0) + 1})

		entropy: float = 0
		n: int = 0
		for value in classes.values():
			n += value

		if n == 0:
			return 1
		p = [i / n for i in classes.values()]

		for i in p:
			entropy += i * math.log2(i)

		return -entropy if entropy != 0 else 0
	
	def predict(self, data: pd.Series):
		if not self.root:
			raise Exception("Model not trained")

		currentNode: DecisionTreeNode | None = self.root
		while currentNode:
			if currentNode.isCNode():
				if data[currentNode.attr] <= currentNode.value:
					currentNode = currentNode.left
					continue
				currentNode = currentNode.right
				continue

			return currentNode.value
		raise Exception("An Unexpected error occurred")

	def __repr__(self):
		return f"{self.root}"

if __name__ == "__main__":
	dt = DecisionTreeClassifier()
	df = pd.DataFrame({
		"x": [0, 0, 1, 1],
		"y": [0, 1, 0, 1]
	}, index=["safe", "safe", "safe", "unsafe"])

	dt.train(df)

	result = dt.predict(pd.Series({"x": 0, "y": 0}))
	print(result)
