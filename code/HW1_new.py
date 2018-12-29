import numpy as np
import pandas as pd
import csv
from collections import defaultdict, namedtuple
import re
import time
import itertools
##########################
### Apriori
##########################
class Apriori(object):
    def __init__(self, min_sup, min_conf=0.0):
        self.min_sup = min_sup
        self.min_conf = min_conf

    def get_freq(self, trans_list, itemset, item_count_dict, min_sup):
        _itemset = set()
        _localset = defaultdict(int) # store temp itemset
        for i in itemset:
            item_count_dict[i] += sum([1 for t in trans_list if i.issubset(t)])
            _localset[i] += sum([1 for t in trans_list if i.issubset(t)])
        for i, ctr in _localset.items():
            _itemset.add(i) if ctr >= min_sup else None

        return _itemset
    
    def get_one_itemset(self, trans_list):
        _itemset = set()
        for l in trans_list:
            for i in l: _itemset.add(frozenset([i]))
        return _itemset

    def fit(self, trans_list):
        itemset = self.get_one_itemset(trans_list)
        item_count_dict = defaultdict(int)
        freq_itemset = dict()
        
        self.trans_len = len(trans_list)
        self.itemset = itemset

        freq_1_term_set = self.get_freq(trans_list, itemset, item_count_dict, self.min_sup)

        # main
        curr_freq = freq_1_term_set
        i = 1
        while curr_freq != set():
            freq_itemset[i] = curr_freq
            i += 1
            curr_cand = set([t1.union(t2) for t1 in curr_freq for t2 in curr_freq if len(t1.union(t2))==i])
            curr_freq = self.get_freq(trans_list, curr_cand, item_count_dict, self.min_sup)
                
        self.item_count_dict = item_count_dict
        self.freq_itemset = freq_itemset

        return item_count_dict, freq_itemset

##########################
### FP-Growth
##########################
class FPTree(object):
    Route = namedtuple('Route', 'Head, Tail')
    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = {}
    
    @property
    def root(self): return self._root
    def add(self, itembag):
        pointer = self._root
        for i in itembag:
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
            child._parent = self

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

def find_frequent_items(itembags, min_support):
    data_dict = defaultdict(lambda: 0)
    # Load items and count the support
    for itembag in itembags:
        for item in itembag: data_dict[item] += 1
    
    # Remove infrequent items from the item support dictionary
    data_dict = dict((item, support) for item, support in data_dict.items() if support >= min_support)

    def sorting(items):
        items = sorted((v for v in items if v in data_dict), reverse=True, key=lambda v: (data_dict[v], v))
        return items
    
    masterTree = FPTree()
    for t in map(sorting, itembags): masterTree.add(t)
    #masterTree.inspect()
    
    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= min_support and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support)
                
                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set): yield s
    
    # Search for frequent itemsets, and yield the results we find.
    for itemset in find_with_suffix(masterTree, []): yield itemset

##########################
### Hash Tree
##########################
class HashNode:
    def __init__(self):
        self.children = {}
        self.isLeaf = True
        self.bucket = {}

class HashTree:
    def __init__(self, max_leaf_size, max_child_size):
        self.root = HashNode()
        self.max_leaf_size = max_leaf_size
        self.max_child_size = max_child_size
        self.frequent_itemsets = {}
    
    def recursive_insert(self, node, itemset, index, count):
        # Recursively add nodes into the tree
        if index == len(itemset):
            if itemset in node.bucket: node.bucket[itemset] += count
            else: node.bucket[itemset] = count
            return
        
        if node.isLeaf:
            if itemset in node.bucket: node.bucket[itemset] += count
            else: node.bucket[itemset] = count
            
            if len(node.bucket) == self.max_leaf_size:
                # bucket has reached the max capacity, need to split
                for old_set, old_count in node.bucket.items():
                    hash_key = self.hash(old_set[index])
                    if hash_key not in node.children:
                        node.children[hash_key] = HashNode()
                    self.recursive_insert(node.children[hash_key], old_set, index+1, old_count)
                del node.bucket
                node.isLeaf = False
        
        else: # node is not a leaf
            hash_key = self.hash(itemset[index])
            if hash_key not in node.children:
                node.children[hash_key] = HashNode()
            self.recursive_insert(node.children[hash_key], itemset, index+1, count)

    def insert(self, itemset):
        itemset = tuple(itemset)
        self.recursive_insert(self.root, itemset, 0, 0)
    
    def add_support(self, itemset):
        node = self.root
        itemset = tuple(sorted(itemset, reverse=True))
        index = 0
        while True:
            if node.isLeaf:
                if itemset in node.bucket: node.bucket[itemset] += 1
                break
            hash_key = self.hash(itemset[index])
            if hash_key in node.children:
                node = node.children[hash_key]
            else: break
            index += 1

    def get_freq(self, min_sup):
        self.frequent_itemsets = []
        self.dfs(self.root, min_sup)
        return self.frequent_itemsets

    def dfs(self, node, min_sup):
        if node.isLeaf:
            for k, v in node.bucket.items():
                if v >= min_sup:
                    self.frequent_itemsets.append((list(k), v))
            return
        for ch in node.children.values(): self.dfs(ch, min_sup)

    def hash(self, key):
        sum = 0
        for char in key:
            sum += ord(char)
            sum2 = "%04d" % (sum)
            digits = int((sum2[2]) + (sum2[3]))
            number = int(digits * digits)
        return number % self.max_child_size
##########################
class Apriori_HashTree(object):
    def __init__(self, min_sup, min_conf=0.0):
        self.min_sup = min_sup
        self.min_conf = min_conf
    
    def get_freq(self, trans_list, itemset, item_count_dict, min_sup):
        _itemset = set()
        _localset = defaultdict(int) # store temp itemset
        for i in itemset:
            item_count_dict[i] += sum([1 for t in trans_list if i.issubset(t)])
            _localset[i] += sum([1 for t in trans_list if i.issubset(t)])
        for i, ctr in _localset.items():
            _itemset.add(i) if ctr >= min_sup else None
        
        return _itemset

    def fit(self, trans_list):
        freq_term_set = find_frequent_one(trans_list, self.min_sup)
        freq_prev = [item[0] for item in freq_term_set]
        i = 2
        while len(freq_prev) > 1:
            new_cand = []
            for k in range(len(freq_prev)):
                j = k+1
                while j < len(freq_prev) and is_prefix(freq_prev[k], freq_prev[j]):
                    new_cand.append(freq_prev[k][:-1] +[freq_prev[k][-1]] + [freq_prev[j][-1]])
                    j += 1
            htree = create_hash_tree(sorted(new_cand, reverse=True), i)
            subsets = generate_k_subsets(trans_list, i)
            for sub in subsets: htree.add_support(sub)
    
            new_freq = htree.get_freq(self.min_sup)
            freq_term_set.extend(new_freq)
            freq_prev = [t[0] for t in new_freq]
            freq_prev.sort(reverse=True)
            i += 1

        return freq_term_set
##########################
def create_hash_tree(itemset, length, max_leaf_size=5, max_child_size=5):
    htree = HashTree(max_child_size, max_leaf_size)
    for item in itemset:
        htree.insert(sorted(item, reverse=True))
    return htree

def find_frequent_one(data_set, support):
    cand_one = {}
    for row in data_set:
        for val in row:
            if val in cand_one: cand_one[val] += 1
            else: cand_one[val] = 1
    frequent_1 = []
    for key, cnt in cand_one.items():
        if cnt >= support: frequent_1.append(([key], cnt))
    return sorted(frequent_1, reverse=True, key=lambda x:(x[1], x[0]))

def generate_k_subsets(dataset, length):
    subsets = []
    for itemset in dataset:
        subsets.extend(map(list, itertools.combinations(itemset, length)))
    return subsets

def is_prefix(list_1, list_2):
    for i in range(len(list_1) - 1):
        if list_1[i] != list_2[i]: return False
    return True
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
survived_title = ['Survived', 'Not_Survived', 'Not_sure_if_Survived']
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
sex_title = ['Male', 'Female', 'Not_Sure_Sex']
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
embark_title = ['Embarked_at_C', 'Embarked_at_S', 'Embarked_at_Q', 'Not_Sure_Where_the_Passenger_Embarked_at']
for i, s in np.ndenumerate(data_embarked):
	index = -1
	if(s=='C'):index = 0
	elif(s=='S'):index = 1
	elif(s=='Q'):index = 2
	item_bag[i[0]].append(embark_title[index])

'''
# Writing edited data to csv
with open('../dataset/Titanic/edited_'+data_name_list[0].split('/')[-1], 'w') as f:	
	writer = csv.writer(f)
	writer.writerows(item_bag)
'''
itemlist = [item_bag]
min_support = [int(0.2*len(titanic))]
###########################
# Dataset2 : IBM Quest Data Generator dataset (w/ [nitems_0.1, ntrans_0.1], [nitems_0.1, ntrans_1], [nitems_1, ntrans_0.1])
for i in [0.1, 1]:
	for j in [0.1, 1]:
		if i == 1 and j == 1: continue
		data_name_list.append('../dataset/ibm/data.nitems_{}.ntrans_{}'.format(i, j))

sup = [0.15, 0.1, 0.05]
for i in range(1, len(data_name_list)):
	item_bag = [[None]]
	ibm_data = open(data_name_list[i], 'r')
	counter = 0
	'''
        # Write csv
        with open('../dataset/ibm/'+data_name_list[i].split('/')[-1]+'.csv', 'w') as f:
            writer = csv.writer(f)
            for line in ibm_data.readlines():
                item = re.split(r'[\s:]+',line.strip(' ').strip('\n'))
                if item_bag == [[None]]:
                    item_bag = [[item[-1]]]
                elif int(float(item[0])) == counter+1:
                    item_bag[counter].append(item[-1])
                else: # New Transaction
                    writer.writerow(item_bag[counter])
                    item_bag.append([item[-1]])
                    counter += 1
            writer.writerow(item_bag[counter])
    '''
	for line in ibm_data.readlines():
		item = re.split(r'[\s:]+',line.strip(' ').strip('\n'))
		if item_bag == [[None]]: item_bag = [[item[-1]]]
		elif int(float(item[0])) == counter+1: item_bag[counter].append(item[-1])
		else: # New Transaction
		    item_bag.append([item[-1]])
		    counter += 1

	itemlist.append(item_bag)
	min_support.append(int(sup[i-1]*counter))
###########################
if __name__ == '__main__':
    for i in range(len(data_name_list)):
        print('Dataset {}: {} w/ min_support = {}'.format(i+1, data_name_list[i].split('/')[-1], min_support[i]))
        ### Apriori
        print('>> Apriori')
        startTime = time.time()
        apri = Apriori(min_support[i])
        item_count_dict, freq_itemset = apri.fit(itemlist[i])
        pat = [[] for i in range(max(freq_itemset.keys()))]
        for key, value in freq_itemset.items():
            print('Frequent {}-term set with min_support({}):\n[\'item\'] count'.format(key, min_support[i]))
            print('-'*30)
            for item in value: pat[int(key)-1].append((list(item), item_count_dict[item]))
            pat[int(key)-1] = sorted(pat[int(key)-1], reverse=True, key=lambda x:(x[1], x[0]))
            for item in pat[int(key)-1]: print('{} :  {}'.format(item[0], item[1]))
            print()

        # Time cost
        spent_time = time.time()-startTime
        print('#'*30)
        print('Spent Time:{} s'.format(spent_time))
        print('#'*30)
        
        # Write to txt
        with open('../results/dataset_{}_{}.txt'.format(i+1, 'apriori'), 'w') as f:
            for key, value in freq_itemset.items():
                f.write('Frequent {}-term set with min_support({}):\n[\'item\'] count\n'.format(key, min_support[i]))
                f.write('-'*30)
                f.write('\n')
                for item in pat[int(key)-1]:
                    f.write('{} :  {}\n'.format(item[0], item[1]))
                f.write('\n')

            f.write('\n')
            f.write('#'*30)
            f.write('\nSpent Time:{} s\n'.format(spent_time))
            f.write('#'*30)
        ###########################
        ### Hash Tree
        print('\n>> Apriori(Hash Tree)')
        startTime = time.time()
        hash = Apriori_HashTree(min_support[i])
        results = hash.fit(itemlist[i])
        h_freq_itemset = sorted(results, reverse= True, key=lambda d: (d[1], d[0]))
        print('Frequent Patterns whose appearance is >= min_support({}):'.format(min_support[i]))
        for key, value in h_freq_itemset:
            print('{} : {}'.format(key, value))

        # Time cost
        spent_time = time.time()-startTime
        print('#'*30)
        print('Spent Time:{} s'.format(spent_time))
        print('#'*30)
        
        # Write to txt
        with open('../results/dataset_{}_{}.txt'.format(i+1, 'hash_tree'), 'w') as f:
            f.write('Frequent Patterns whose appearance is >= min_support({}):\n'.format(min_support[i]))
            for key, value in h_freq_itemset: f.write('  {} :  {}\n'.format(key, value))

            f.write('\n')
            f.write('#'*30)
            f.write('\nSpent Time:{} s\n'.format(spent_time))
            f.write('#'*30)
        ###########################
        ### FP-Growth
        print('\n>> FP-Growth')
        startTime = time.time()
        results = [(itemset, support) for itemset, support in find_frequent_items(itemlist[i], min_support[i])]
        fp_freq_itemset = sorted((v for v in results),reverse = True, key = lambda v: (v[1],v[0]))
        for key, value in fp_freq_itemset: print('{} : {}'.format(key, value))
        print()

        # Time cost
        spent_time = time.time()-startTime
        print('#'*30)
        print('Spent Time:{} s'.format(spent_time))
        print('#'*30)

        # Write to txt
        with open('../results/dataset_{}_{}.txt'.format(i+1, 'fpgrowth'), 'w') as f:
            f.write('Frequent Patterns whose appearance is >= min_support({}):\n'.format(min_support[i]))
            for key, value in fp_freq_itemset: f.write('  {} :  {}\n'.format(key, value))

            f.write('\n')
            f.write('#'*30)
            f.write('\nSpent Time:{} s\n'.format(spent_time))
            f.write('#'*30)
        ###########################
