import os
import sys
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict, Counter

import sklearn
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.layers import Dense

import stellargraph as sg
from stellargraph import globalvar
from stellargraph.data import EdgeSplitter, UniformRandomWalk, UnsupervisedSampler
from stellargraph.mapper import RelationalFullBatchNodeGenerator, FullBatchNodeGenerator, GraphSAGELinkGenerator, GraphSAGENodeGenerator, HinSAGELinkGenerator
from stellargraph.layer import RGCN, GCN, GAT, GraphSAGE, link_classification, HinSAGE, link_regression


def main():
	#import vertices and edges data
	v_data = pd.read_csv("polimi.case.graphs.vertices.csv", sep=',', index_col='node_id')
	e_data = pd.read_csv("polimi.case.graphs.edges.csv", sep=',', index_col='edge_id')
	#import core and extended targets for training
	core_target = pd.read_csv("training.core.vertices.csv", sep='\t', index_col='NodeID')      
	ext_target = pd.read_csv("training.extended.vertices.csv", sep='\t', index_col='NodeID')

	#'''
	#subsample edges
	e_sample = e_data.sample(n=3000)
	#gather vertices from edges
	v_list = list(pd.Categorical(list(e_sample.to_id)+list(e_sample.from_id)).categories)
	#subsample vertices
	v_sample = v_data[v_data.index.isin(v_list)]
	#subsample targets
	core_target_sample = core_target[core_target.index.isin(v_list)] 
	ext_target_sample = ext_target[ext_target.index.isin(v_list)] 
	'''
	#comment previous part and uncomment the following lines to employ the whole graph
	e_sample = e_data
	v_sample = v_data
	core_target_sample = core_target
	ext_target_sample = ext_target
	'''
	#TBD with Oracle
	#create validation set (used only for final performance measurement) and remove it from dataset
	#v_valid = v_sample[v_sample.testingFlag == 1]
	#v_sample = v_sample[v_sample.testingFlag != 1]
	#e_sample = e_sample.drop()
		
	#set missing Core/Extended ID to 0 -> could be a way, but other solutions are appreciated
	v_sample.CoreCaseGraphID = v_sample.CoreCaseGraphID.fillna(0)
	v_sample.ExtendedCaseGraphID = v_sample.ExtendedCaseGraphID.fillna(0) 

	'''
	#create dataframes for each kynd of node (['Account', 'Address', 'Customer', 'Derived Entity', 'External Entity'])
	#In a dumb but explicable way
	# Account: 'Revenue Size Flag', 'Account ID String', 'CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'
	v_account = v_sample[v_sample.Label == 'Account']
	v_account = v_account.drop(['Label', 'Address', 'Person or Organisation', 'Name', 'Income Size Flag'], axis=1)
	# Address: 'Address', 'CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'
	v_address = v_sample[v_sample.Label == 'Address']
	v_address = v_address.drop(['Label', 'Revenue Size Flag', 'Account ID String', 'Person or Organisation', 'Name', 'Income Size Flag'], axis=1)
	# Customer: 'Person or Organisation', 'Name', 'Income Size Flag', 'CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'
	v_customer = v_sample[v_sample.Label == 'Customer']
	v_customer = v_customer.drop(['Label', 'Revenue Size Flag', 'Account ID String', 'Address'], axis=1)
	# Derived Entity: 'Person or Organisation', 'Name', 'CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'
	v_der_ent = v_sample[v_sample.Label == 'Derived Entity']
	v_der_ent = v_der_ent.drop(['Label', 'Revenue Size Flag', 'Account ID String', 'Address', 'Income Size Flag'], axis=1)
	# External Entity: 'Person or Organisation', 'Name', 'CoreCaseGraphID', 'ExtendedCaseGraphID', 'testingFlag'
	v_ext_ent = v_sample[v_sample.Label == 'Derived Entity']
	v_ext_ent = v_ext_ent.drop(['Label', 'Revenue Size Flag', 'Account ID String', 'Address', 'Income Size Flag'], axis=1)
	'''
	
	#create dataframes for each kynd of node (['Account', 'Address', 'Customer', 'Derived Entity', 'External Entity'])
	#In a smart way:
	v_sets = defaultdict()
	for v_type in list(pd.Categorical(v_sample.Label).categories):
		v_sets[v_type] = v_sample[v_sample.Label == v_type]
		v_sets[v_type] = v_sets[v_type].drop(['Label', 'testingFlag']+list(v_sets[v_type].columns[v_sets[v_type].isnull().all()]), axis=1)

	#create dataframes for each kynd of edge (['has account', 'has address', 'is similar', 'money transfer'])
	#In a smart way:
	e_sets = defaultdict()
	for e_type in list(pd.Categorical(e_sample.Label).categories):
		e_sets[e_type] = e_sample[e_sample.Label == e_type]
		e_sets[e_type] = e_sets[e_type].drop(['Label']+list(e_sets[e_type].columns[e_sets[e_type].isnull().all()]), axis=1)
		e_sets[e_type] = e_sets[e_type].rename(columns={'from_id':'source', 'to_id':'target'})

	#convert non numerical data in numerical data
	#1. "logical" conversion
	#Revenue Size Flag: low, mid_low, medium, mid_high, high -> 1,2,3,4,5
	conversion = {'low':1, 'mid_low':2, 'medium':3, 'mid_high':4, 'high':5}
	for i in v_sets:
		if 'Revenue Size Flag' in list(v_sets[i].columns):
			v_sets[i]['Revenue Size Flag']=v_sets[i]['Revenue Size Flag'].map(conversion)
	#Income Size Flag: low, medium, high -> 1,2,3
	conversion = {'low':1, 'medium':2, 'high':3}
	for i in v_sets:
		if 'Income Size Flag' in list(v_sets[i].columns):
			v_sets[i]['Income Size Flag']=v_sets[i]['Income Size Flag'].map(conversion)
	#Similarity Strength: weak, medium, strong -> 1,2,3
	conversion = {'weak':1, 'medium':2, 'strong':3}
	for i in e_sets:
		if 'Similarity Strength' in list(e_sets[i].columns):
			e_sets[i]['Similarity Strength']=e_sets[i]['Similarity Strength'].map(conversion)
	#Amount Flag: small, medium, large -> 10,100,1000 (just to change the logic, the final choice is up to you) -> treated as weights
	conversion = {'small':10, 'medium':100, 'large':1000}
	for i in e_sets:
		if 'Amount Flag' in list(e_sets[i].columns):
			e_sets[i]['Amount Flag']=e_sets[i]['Amount Flag'].map(conversion)
			e_sets[i] = e_sets[i].rename(columns={'Amount Flag':'weight'})

	#2. one-hot encoding
	#Person or Organisation: create 2 bool columns, one for Person, one for Organisation (could have just created a single boolean column: 0->Person, 1->Organization)
	for i in v_sets:
		if 'Person or Organisation' in list(v_sets[i].columns):
			v_sets[i] = pd.get_dummies(v_sets[i], columns=['Person or Organisation'])

	#3. more complex transformations (i.e. from strings to numbers) -> the limit is your imagination!
	#Vertices: Account ID String, Address, Name
	#Fast solution: dropping non numerical attributes, but we're loosing lot of information
	for i in v_sets:
		if 'Account ID String' in list(v_sets[i].columns):
			v_sets[i] = v_sets[i].drop('Account ID String', axis=1)
		if 'Address' in list(v_sets[i].columns):
			v_sets[i] = v_sets[i].drop('Address', axis=1)
		if 'Name' in list(v_sets[i].columns):
			v_sets[i] = v_sets[i].drop('Name', axis=1)

	# Create Graph in stellargraph 
	#G = sg.StellarGraph(v_sets, e_sets)
	# Create Directed Graph in stellargraph 
	G = sg.StellarDiGraph(v_sets, e_sets)

	# Print info about the graph we just built
	print(G.info())


#### Graph embedding with NODE2VEC and WORD2VEC
	rw = sg.data.BiasedRandomWalk(G)
	walks = rw.run(
		nodes=list(G.nodes()),  # root nodes
		length=10,  # maximum length of a random walk
		n=10,  # number of random walks per root node
		p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
		q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
	)
	print("Number of random walks: {}".format(len(walks)))

	#import Word2Vec model from gensim library
	from gensim.models import Word2Vec
	#convert int ID to str ID, according to gensim library
	str_walks = [[str(n) for n in walk] for walk in walks]
	#train Representation Learning
	model = Word2Vec(str_walks, size=128, window=5, min_count=0, sg=1, workers=8, iter=5)
	# The embedding vectors can be retrieved from model.wv using the node ID.
	#model.wv["19231"].shape 
	
	# Retrieve node embeddings 
	node_ids = model.wv.index2word  # list of node IDs
	node_embeddings = (model.wv.vectors)  # numpy.ndarray of size number of nodes times embeddings dimensionality
	
	# Retrieve corresponding targets
	# from training csv
	#core_targets = core_target_sample.loc[[int(node_id) for node_id in node_ids if int(node_id) in list(core_target_sample.index)]].CaseID
	#ext_targets = ext_target_sample.loc[[int(node_id) for node_id in node_ids if int(node_id) in list(ext_target_sample.index)]].CaseID
	# from vertices' data
	core_targets = v_sample.loc[[int(node_id) for node_id in node_ids]].CoreCaseGraphID
	ext_targets = v_sample.loc[[int(node_id) for node_id in node_ids]].ExtendedCaseGraphID
	
	# Transform the embeddings to 2d space for visualization
	transform = TSNE #PCA 
	trans = transform(n_components=2)
	node_embeddings_2d = trans.fit_transform(node_embeddings)

	# draw the embedding points, coloring them by the target label (CaseID)
	alpha = 0.7
	label_map = {l: i for i, l in enumerate(np.unique(ext_targets), start=10) if pd.notna(l)}
	node_colours = [label_map[target] if pd.notna(target) else 0 for target in ext_targets]

	plt.figure(figsize=(7, 7))
	plt.axes().set(aspect="equal")
	plt.scatter(
		node_embeddings_2d[:, 0],
		node_embeddings_2d[:, 1],
		c=node_colours,
		cmap="jet",
		alpha=alpha,
	)
	plt.title("{} visualization of node embeddings w.r.t. Extended Case ID".format(transform.__name__))
	plt.show()


#### Node classification with GraphSAGE, GCN, GAT

	# Split in train(70%), test(15%) and validation set (15%) 
	train_ID, test_ID = model_selection.train_test_split(
		ext_targets, train_size=0.7, test_size=None, #stratify=ext_targets
	)
	val_ID, test_ID = model_selection.train_test_split(
		test_ID, train_size=0.5, test_size=None, #stratify=test_ID
	)
	len(train_ID.index, val_ID.index, test_ID.index)

	# Convert targets labels in one-hot encoded features (optional, for categorical targets)
	target_encoding = preprocessing.LabelBinarizer()
	train_targets = target_encoding.fit_transform(train_ID)
	val_targets = target_encoding.transform(val_ID)
	test_targets = target_encoding.transform(test_ID)

	model_type = "graphsage" # gcn, gat
	use_bagging = (True)

	if model_type == "graphsage":
		# For GraphSAGE model
		batch_size = 50
		num_samples = [10, 10]
		n_estimators = 5  # The number of estimators in the ensemble
		n_predictions = 10  # The number of predictions per estimator per query point
		epochs = 50  # The number of training epochs
	elif model_type == "gcn":
		# For GCN model
		n_estimators = 5  # The number of estimators in the ensemble
		n_predictions = 10  # The number of predictions per estimator per query point
		epochs = 50  # The number of training epochs
	elif model_type == "gat":
		# For GAT model
		layer_sizes = [8, train_targets.shape[1]]
		attention_heads = 8
		n_estimators = 5  # The number of estimators in the ensemble
		n_predictions = 10  # The number of predictions per estimator per query point
		epochs = 200  # The number of training epochs

	if model_type == "graphsage":
		generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
		train_gen = generator.flow(train_ID.index, train_targets, shuffle=True)
	elif model_type == "gcn":
		generator = FullBatchNodeGenerator(G, method="gcn")
		train_gen = generator.flow(train_ID.index, train_targets)  # does not support shuffle
	elif model_type == "gat":
		generator = FullBatchNodeGenerator(G, method="gat")
		train_gen = generator.flow(train_ID.index, train_targets)  # does not support shuffle 

	if model_type == "graphsage":
		base_model = GraphSAGE(
			layer_sizes=[16, 16],
			generator=generator,
			bias=True,
			dropout=0.5,
			normalize="l2"
		)
		x_inp, x_out = base_model.in_out_tensors()
		predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
	elif model_type == "gcn":
		base_model = GCN(
			layer_sizes=[32, train_targets.shape[1]],
			generator=generator,
			bias=True,
			dropout=0.5,
			activations=["relu", "softmax"],
		)
		x_inp, predictions = base_model.in_out_tensors()
	elif model_type == "gat":
		base_model = GAT(
			layer_sizes=layer_sizes,
			attn_heads=attention_heads,
			generator=generator,
			bias=True,
			in_dropout=0.5,
			attn_dropout=0.5,
			activations=["relu", "softmax"],
		)
		x_inp, predictions = base_model.in_out_tensors()
	
	model = Model(inputs=x_inp, outputs=predictions)
	
	if use_bagging:
		model = BaggingEnsemble(model, n_estimators=n_estimators, n_predictions=n_predictions)
	else:
		model = Ensemble(model, n_estimators=n_estimators, n_predictions=n_predictions)

	model.compile(
		optimizer=optimizers.Adam(lr=0.005),
		loss=losses.categorical_crossentropy,
		metrics=["acc"],
	)

	val_gen = generator.flow(val_ID.index, val_targets)
	test_gen = generator.flow(test_ID.index, test_targets)
	
	if use_bagging:
		# When using bootstrap samples to train each model in the ensemble, we must specify
		# the IDs of the training nodes (train_data) and their corresponding target values
		# (train_targets)
		history = model.fit(
			generator,
			train_data=train_gen.index,
			train_targets=train_targets,
			epochs=epochs,
			validation_data=val_gen,
			verbose=0,
			shuffle=False,
			bag_size=None,
			use_early_stopping=True,  # Enable early stopping
			early_stopping_monitor="val_acc",
		)
	else:
		history = model.fit(
			train_gen,
			epochs=epochs,
			validation_data=val_gen,
			verbose=0,
			shuffle=False,
			use_early_stopping=True,  # Enable early stopping
			early_stopping_monitor="val_acc",
		)

	sg.utils.plot_history(history)

	test_metrics = model.evaluate(test_gen)
	print("\nTest Set Metrics:")
	for name, val in zip(model.metrics_names, test_metrics):
		print("\t{}: {:0.4f}".format(name, val))


if __name__ == "__main__":
	main()