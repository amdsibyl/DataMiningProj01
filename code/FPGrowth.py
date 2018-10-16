import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
import re

class FPTree(object):
	Route = namedtuple('Route', 'Head, Tail')
	def __init__(self):
		self._root = FPNode(self, None, None)
		self._routes = {}

	@property
	def root(self): return self._root
	def add(self, transaction):
		pointer = self._root
		for i in transaction:
			next_pointer = pointer.search(i)
			if next_pointer: next_pointer.increment_count()
			else:
				next_pointer = FPNode(self, i)
				pointer.add(next_pointer)
				self.update_route(next_pointer)
			pointer = next_pointer

	def update_route(self, p):
		assert self is p.tree
		try:
			route = self._routes[p.item]
			route[1].nodeLink = p
			self._routes[p.item] = self.Route(route[0], p)
		except KeyError:
			self._routes[p.item] = self.Route(p, p)

	def items(self):
		for item in self._routes.keys():
			yield (item, self.nodes(item))
	def nodes(self, item):
		try: node = self._routes[item][0]
		except KeyError: return
		while node:
			yield node
			node = node.nodeLink

	def prefix_paths(self, item):
		def collect_paths(node):
			path=[]
			while node and not node.root:
				path.append(node)
				node = node.parent
			path.reverse()
			return path
		return (collect_paths(node) for node in self.nodes(item))

	def inspect(self):
		print('Tree:')
		self.root.inspect(1)
		print('\nRoutes:')
		for item, nodes in self.items():
			print('\t%r' %item)
			for node in nodes: 
				print('\t\t-%r' %node)

class FPNode(object):
	def __init__(self, tree, item, count=1):
		self._tree = tree
		self._item = item
		self._count = count
		self._nodeLink = None
		self._parent = None
		self._children = {}

	def add(self, child):
		if not isinstance(child, FPNode):
			raise TypeError("Only FPNodes can be added as children")
		if not child.item in self._children:
			self._children[child.item] = child
			child.parent = self

	def search(self, item):
		try: return self._children[item]
		except KeyError: return None

	def increment_count(self):
		if self._count is None:
			raise ValueError("Roots nodes have no count records")
		self._count += 1	

	def __contains__(self, item):
		return item in self._children

	@property
	def tree(self): return self._tree
	@property
	def item(self): return self._item
	@property
	def count(self): return self._count
	@property
	def root(self): return self._item is None and self._count is None
	@property
	def leaf(self): return len(self._children) == 0
	@property
	def parent(self): return self._parent
	@property
	def nodeLink(self): return self._nodeLink
	@property
	def children(self): return tuple(self._children.values())

	@parent.setter
	def parent(self, value):
		if value is not None and not isinstance(value, FPNode):
			raise TypeError("A node must have an FPNode as a parent")
		if value and value.tree is not self.tree:
			raise ValueError("Shouldn't exist a parent from another tree")
		self._parent = value	

	@nodeLink.setter
	def nodeLink(self, value):
		if value is not None and not isinstance(value, FPNode):
			raise TypeError("A node must have an FPNode as a linked node")
		if value and value.tree is not self.tree:
			raise ValueError("Shouldn't exist a linked node from another tree")
		self._nodeLink = value	

	def inspect(self, depth=0):
		print (('\t' * (depth-1)) + repr(self))
		for child in self.children: child.inspect(depth + 1)

	def __repr__(self):
		if self.root: return ("<%s (root)>" % type(self).__name__)
		return ("<%s %r (%r)>" % (type(self).__name__, self.item, self.count))

def conditional_tree_from_paths(paths):
	tree = FPTree()
	cond_item = None
	items = set()
	# Counts will be reconstructed from the counts of leaf nodes
	for p in paths:
		if cond_item is None: cond_item = p[-1].item
		pointer = tree.root
		for node in p:
			next_pointer = pointer.search(node.item)
			if not next_pointer:
				items.add(node.item)
				count = node.count if node.item == cond_item else 0
				next_pointer = FPNode(tree, node.item, count)
				pointer.add(next_pointer)
				tree.update_route(next_pointer)
			pointer = next_pointer	
	assert cond_item is not None

	# Calculate counts of non-leaf nodes
	for p in tree.prefix_paths(cond_item):
		count = p[-1].count
		for node in reversed(p[:-1]): node._count += count

	return tree

def find_frequent_items(itembags, min_support, include_support=False):
	data_dict = defaultdict(lambda: 0) # mapping from items to their supports
	# Load items and count the support
	for itembag in itembags:
		for item in itembag: data_dict[item] += 1

	# Remove infrequent items from the item support dictionary.
	data_dict = dict((item, support) for item, support in data_dict.items() if support >= min_support)

	def sorting_transaction(items):
		items = sorted((v for v in items if v in data_dict), reverse=True, key=lambda v: data_dict[v])
		return items

	masterTree = FPTree()
	for t in map(sorting_transaction, itembags): masterTree.add(t)
	masterTree.inspect()

	def find_with_suffix(tree, suffix):
		for item, nodes in tree.items():
			support = sum(n.count for n in nodes)
			if support >= min_support and item not in suffix:
				found_set = [item] + suffix
				yield (found_set, support) if include_support else found_set

				cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
				find_with_suffix(cond_tree, found_set)
				for s in find_with_suffix(cond_tree, found_set): yield s

	# Search for frequent itemsets, and yield the results we find.
	for itemset in find_with_suffix(masterTree, []): yield itemset

##########################
# Dataset1 : Titanic:Machine Learning from Disaster from Kaggle
titanic_file = "../dataset/Titanic/train.csv"
titanic = pd.read_csv(titanic_file)
data_name_list = [titanic_file]
# Columns : PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# Delete the data columns that aren't important for analyzing
titanic.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin"], axis=1, inplace=True)
item_bag = [[None]] * len(titanic)
# Columns : Survived, Pclass, Sex, Age, Embarked
# Survived indexing
data_survived = np.array(titanic.iloc[:, 0], int)
survived_title = ['Survived', 'Not Survived', 'Not Sure if Survived']
for i, s in np.ndenumerate(data_survived):
	index = -1
	if s==1: index = 0
	elif s==0: index = 1
	item_bag[i[0]] = [survived_title[index]]

# Pclass indexing
data_pclass = np.array(titanic.iloc[:, 1], int)
pclass_title = ['Pclass1', 'Pclass2', 'Pclass3']
for i, s in np.ndenumerate(data_pclass):
	index = -1
	if s==1 or s==2 or s==3: index = s-1
	item_bag[i[0]].append(pclass_title[index])

# Sex indexing
data_sex = np.array(titanic.iloc[:, 2])
sex_title = ['Male', 'Female', 'Not Sure Sex']
for i, s in np.ndenumerate(data_sex):
	index = -1
	if s=="male": index = 0 
	elif s=="female": index = 1
	item_bag[i[0]].append(sex_title[index])

# Age range indexing
data_age = np.array(titanic.iloc[:, 3], int)
age_range = ['Kids(0~8)', 'Teenager(9~18)', 'Adult(19~40)', 'Middle(41~65)', 'Elder(66~)']
for i, s in np.ndenumerate(data_age):
	index = -1
	if s<=8: index = 0
	elif s<=18 : index = 1
	elif s<=40 : index = 2
	elif s<=65 : index = 3
	elif s>65 : index = 4
	item_bag[i[0]].append(age_range[index])

# Embarked : C / S / Q
data_embarked = np.array(titanic.iloc[:, 4])
embark_title = ['Embark at C', 'Embark at S', 'Embark at Q', 'Not Sure Where the Passenger Embark at']
for i, s in np.ndenumerate(data_embarked):
	index = -1
	if(s=='C'):index = 0
	elif(s=='S'):index = 1
	elif(s=='Q'):index = 2
	item_bag[i[0]].append(embark_title[index])

itemlist = [item_bag]
min_support = [200]
###########################
# Dataset2 : IBM Quest Data Generator dataset (w/ [nitems_0.1, ntrans_0.1], [nitems_0.1, ntrans_1], [nitems_1, ntrans_0.1])
for i in [0.1, 1]:
	for j in [0.1, 1]:
		if i == 1 and j == 1: continue
		data_name_list.append('../dataset/ibm/data.nitems_{}.ntrans_{}'.format(i, j))

sup = [15, 100, 5]
for i in range(1, len(data_name_list)):
	item_bag = [[None]]
	ibm_data = open(data_name_list[i], 'r')
	counter = 0
	for line in ibm_data.readlines():
		item = re.split(r'[\s:]+',line.strip(' ').strip('\n'))
		if item_bag == [[None]]:
			item_bag = [[item[-1]]]
		elif int(float(item[0])) == counter+1:
			item_bag[counter].append(item[-1])
		else:
			item_bag.append([item[-1]])
			counter += 1

	itemlist.append(item_bag)
	min_support.append(sup[i-1])

### test data ###
"""
data = [
['Milk', 'Bread', 'Beer'],
['Bread', 'Coffee'],
['Bread', 'Egg'],
['Milk', 'Bread', 'Coffee'],
['Milk', 'Egg'],
['Bread', 'Egg'],
['Milk', 'Egg'],
['Milk', 'Bread', 'Egg', 'Beer'],
['Milk', 'Bread', 'Egg']
]
item_bag = data
min_support = 3
"""
###########################
for i in range(len(data_name_list)):
	print('Dataset {}: {} w/ min_support = {}'.format(i+1, data_name_list[i].split('/')[-1], min_support[i]))
	results = []
	for itemset, support in find_frequent_items(itemlist[i], min_support[i], True):
		results.append((itemset, support))
	results = sorted((v for v in results),reverse = True, key = lambda v: v[1])
	print('\nFrequent Patterns whose appearance is >= min_support({}): '.format(min_support[i]))
	for itemset, support in results:
		print('\t{} {}'.format(itemset,support))
	print('###########################')

