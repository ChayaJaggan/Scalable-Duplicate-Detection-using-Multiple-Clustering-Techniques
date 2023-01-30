import sys

import Functions
from Functions import *
sys.path.insert(0, 'C:/Users/chaya/PycharmProjects/pythonProject/Computer Science')
from statistics import mean, stdev
# run this file to obtain the results
import seaborn as sns
import matplotlib.pyplot as plt


for x in range(1, 6):
	print("Results run: ", x)
	#def executeMain():
	results = pd.DataFrame({
			'Pair Quality': [],
			'Pair Completeness': [],
			'Fraction Comparisons': [],
			'F1_star': [],
			'F1_single': [],
			'epsilon': [],
			'F1_complete': [],
			'F1_DBSCAN': [],
		})

	""" Load the data """
	with open('C:/Users/chaya/PycharmProjects/pythonProject/Computer Science/TVs-all-merged.json', 'r') as f:
		data = f.read()
		for line in f:
			content = line.strip().split()
			print(content)
	# dictionary of data, only unique items
	dict = json.loads(data)

	dictList = []
	for key, value in dict.items():
		for value in dict[key]:
			dictList.append(key)
			dictList.append(value)
	# list of only the product discriptions
	productDescriptions = dictList[1::2]

	""" Bootstrapping """
	train,test  = Functions.bootstraps(productDescriptions)

	# training data
	for value in train['train'].values():
		train_items = [item for item in value]

	def train(train_items):
		# titles
		titles = Functions.getTitle(train_items)
		processedTitle = Functions.preProcess(titles)
		dictionary = Functions.getDictionary(processedTitle)
		binaryVector = Functions.getBinaryVector(dictionary,train_items)
		signatureMatrix = Functions.minhash(binaryVector,prime = 1423)
		maximum = []
		for b in [1,20,30,60,70,80,90,100,110,120,130,140,150,200,210,230,300]:
			print("b is equal to " + str(b))
			F1 = {}
			F1_complete = {}
			LSH, candidates = Functions.localitySensitiveHashing(signatureMatrix, binaryVector,b)
			#print(len(LSH))
			PQ = Functions.pairQuality(candidates,train_items)
			PC = Functions.pairCompleteness(candidates,train_items)
			F1_star = Functions.starmeasure(PQ, PC)
			print("The measures F1_star, PQ, and PC are " + str(F1_star) + " " + str(PQ) + " " + str(PC))
			dissimilarityMatrix = Functions.dissimilarityMatrix(train_items,LSH,processedTitle)
			for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5,0.55 ,0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95,1.0]:
				cluster = Functions.clustering_single(dissimilarityMatrix, epsilon)
				F1_value = Functions.F1measure(train_items, cluster)
				F1[epsilon] = F1_value
				cluster2 = Functions.clustering_complete(dissimilarityMatrix, epsilon)
				F1_complete_value = Functions.F1measure(train_items, cluster2)
				F1_complete[epsilon] = F1_complete_value
				cluster3 = Functions.clustering_DBSCAN(dissimilarityMatrix, epsilon)
				F1_DBSCAN = Functions.F1measure(train_items, cluster3)
				F1[epsilon] = F1_value
			max_eps = max(F1, key=F1.get)
			maximum.append(max_eps)

		mean_eps = mean(maximum)
		std_eps = stdev(maximum)
		mean_f1 = mean(F1.values())
		print("The best epsilon value is " + str(max_eps))
		return mean_eps, std_eps, mean_f1, F1, F1_complete, F1_DBSCAN


	epsilon, std_eps, mean_F1, F1, F1_complete, F1_DBSCAN = train(train_items)
	print("The training is over, mean, stdev and F1 are " + str(epsilon) + " " + str(std_eps) + " " + str(mean_F1))
	print("The F1 of training is " + str(F1) + " and F1_complete is " + str(F1_complete) + " and F1_DBSCAN is " + str(
		F1_DBSCAN))

	# test data
	for value in test['test'].values():
		test_items = [item for item in value]

	def test(test_items,epsilon):
		titles = Functions.getTitle(test_items)
		processedTitle = Functions.preProcess(titles)
		dictionary = Functions.getDictionary(processedTitle)
		binaryVector = Functions.getBinaryVector(dictionary,test_items)
		signatureMatrix = Functions.minhash(binaryVector,prime = 887)
		results = pd.DataFrame({
			'Pair Quality': [],
			'Pair Completeness': [],
			'Fraction Comparisons': [],
			'F1_star': [],
			'F1_single': [],
			'epsilon': [],
			'F1_complete': [],
			'F1_DBSCAN': [],

		})
		for b in [1, 5, 10, 20, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 200, 210, 220,
				  230, 240, 250, 280, 300]:
			print("b is equal to " + str(b))
			LSH, candidates = Functions.localitySensitiveHashing(signatureMatrix, binaryVector, b)
			PQ = Functions.pairQuality(candidates,test_items)
			PC = Functions.pairCompleteness(candidates,test_items)
			FC = Functions.fracComparisons(candidates,test_items)
			F1_star = Functions.starmeasure(PQ,PC)
			print("The F1_star measure is " + str(F1_star))
			dissimilarityMatrix = Functions.dissimilarityMatrix(test_items,LSH,processedTitle)
			cluster = Functions.clustering_single(dissimilarityMatrix, epsilon)
			F1_single = Functions.F1measure(test_items, cluster)
			cluster2 = Functions.clustering_complete(dissimilarityMatrix, epsilon)
			F1_complete = Functions.F1measure(test_items, cluster2)
			cluster3 = Functions.clustering_DBSCAN(dissimilarityMatrix, epsilon)
			F1_DBSCAN = Functions.F1measure(test_items, cluster3)

			print("The F1 measure is " + str(F1))
			results = results.append({
			'Pair Quality': PQ,
			'Pair Completeness': PC,
			'Fraction Comparisons': FC,
			'F1_star': F1_star,
			'F1_single': F1_single,
			'epsilon': epsilon,
			'F1_complete': F1_complete,
			'F1_DBSCAN': F1_DBSCAN
			}, ignore_index=True)
		return results


	results_test = test(test_items, epsilon)
	print(results_test)

	fig_PQ_test = sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['Pair Quality'])
	plt.savefig('PQ_test' + str(x))
	fig_PQ_test.clear()
	fig_PC_test = sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['Pair Completeness'])
	plt.savefig('PC_test' + str(x))
	fig_PC_test.clear()
	fig_F1_test = sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['F1_single'])
	sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['F1_complete'])
	sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['F1_DBSCAN'])
	plt.savefig('F1_test' + str(x))
	# plt.savefig('F1_single_test' + str(x))
	fig_F1_test.clear()
	# fig_F1_complete_test = sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['F1_complete'])
	# plt.savefig('F1_complete_test' + str(x))
	# fig_F1_complete_test.clear()
	# fig_F1_DBSCAN_test = sns.lineplot(x=results_test['Fraction Comparisons'], y=results_test['F1_DBSCAN'])
	# plt.savefig('F1_DBSCAN_test' + str(x))
	# fig_F1_DBSCAN_test.clear()

	file = open("results" + str(x) + ".txt", "w+")
	# Saving the array in a text file
	content = results_test.to_string(header=True, index=False)
	file.write(content)
	file.close()

	results = results.append(results_test)

	print(results)
	print("The end of " + str(x))

# Pair Quality figure
fig_PQ = sns.lineplot(x=results['Fraction Comparisons'], y=results['Pair Quality'])
plt.savefig('PQ' + str(x))
plt.show()
fig_PQ.clear()
# Pair Completeness figure
fig_PC = sns.lineplot(x=results['Fraction Comparisons'], y=results['Pair Completeness'])
plt.savefig('PC' + str(x))
plt.show()
fig_PC.clear()
# F1 figure
fig_F1 = sns.lineplot(x=results['Fraction Comparisons'], y=results['F1_single'])
plt.savefig('F1_single' + str(x))
plt.show()
fig_F1.clear()
fig_F1_complete = sns.lineplot(x=results['Fraction Comparisons'], y=results['F1_complete'])
plt.savefig('F1_complete' + str(x))
plt.show()
fig_F1_complete.clear()
fig_F1_DBSCAN = sns.lineplot(x=results['Fraction Comparisons'], y=results['F1_DBSCAN'])
plt.savefig('F1_DBSCAN' + str(x))
plt.show()
fig_F1_complete.clear()


# final figure in one
a = results['F1_single']
b = results['F1_complete']
c = results['F1_DBSCAN']
d = results['Fraction Comparisons']
lines = plt.plot(d, a, d, b, d, c)
l1, l2, l3 = lines
plt.setp(lines, linestyle='-')  # set both to dashed
plt.setp(l1, linewidth=2, color='b')  # line1 is thick and red
plt.setp(l2, linewidth=1, color='g')
plt.setp(l3, linewidth=2, color='r')
plt.legend(('Single', 'Complete', 'DBSCAN'), shadow=True)
plt.xlabel('Fraction Comparisons')
plt.ylabel('F1 score')
plt.savefig('F1_final' + str(x))
plt.show()


#def main():
			#executeMain()

	#if __name__ == '__main__':
			#main()
