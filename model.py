import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing

import pdb

import open_subset as op
import parsers as pa


class Classifier:

	def __init__(self):
		
		#self.file_loc = file_loc
		self.data = None
		self.pred = None

		print "ready to open file. Run .open_file().  Type options: simple, feature_subset, transform"

	# open 
	def open_file(self, file_location):
		d = op.open_and_process(file_location, 'transform', 'model', 'train')
		del d['status_Default']
		del d['issue_d']
		self.data = d
		print "run .assign_to_groups()"

	def assign_to_groups(self):
		# separate out predictor
		self.pred = self.data['status_Fully Paid']

		# assign each to bulk groups
		n = 13000
		self.tr_feat = self.data[:n]
		del self.tr_feat['status_Fully Paid']
		self.tr_pred = self.pred[:n]
		self.tst_dt = self.data[n:]

	def prepare(self):
		self.scaler = preprocessing.StandardScaler()
		self.tr_feat = self.scaler.fit_transform(self.tr_feat)
		self.tr_pred = self.tr_pred.values

	def run(self, prelim_test=None ,split=None):
		# train logistic regression model
		self.clf = linear_model.SGDClassifier()
		probas_ = self.clf.fit(self.tr_feat, self.tr_pred)
		# do analysis

		if prelim_test is "yes":
			if split is "by_grades":
				grades = ['A','B','C','D','E','F','G']
				for letter in grades:
					dt = self.tst_dt[self.tst_dt['grade_'+letter] == 1]
					tst_pred = dt['status_Fully Paid']
					tst_feat = dt
					del tst_feat['status_Fully Paid'] # change to remove using .drop : see http://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
					print "Model performance on notes of grade " + letter
					self.single_analysis((self.scaler.transform(tst_feat)), tst_pred)
			else:
				tst_pred = self.tst_dt['status_Fully Paid']
				tst_feat = self.tst_dt
				del tst_feat['status_Fully Paid']
				self.single_analysis(self.scaler.transform(tst_feat), tst_pred)

	def single_analysis(self, features, true_results):
		# general accuracy of model
		print "Accuracy of model: ", self.clf.score(features, true_results)
		# information on total loan set, by grade
		total_paid = sum(true_results)
		total_notes = len(true_results)
		print "Total Fully Paid in Grade: " + str(total_paid)
		print "Total Notes in Grade: " + str(total_notes)
		print "None-default rate if fully invest: " + str(float(total_paid)/total_notes)
		# information on performance of the model
		pr = self.clf.predict(features)
		total_investments_by_prediction = sum(pr)
		print "Total number invested by model: " + str(total_investments_by_prediction)
		################
		ac = true_results.values
		correct = 0
		for i in range(len(pr)):
			if (pr[i] == 1) and (ac[i] == 1):
				correct += 1
		print "non-default rate invested by model: " + str(float(correct)/sum(pr))
		self.display_ROC(pr, ac)

	def display_ROC(self, pr, ac):
		##### make ROC graph for the result #####
		if sum(pr) == len(pr):
			print "NO ROC : model invested in all notes"
			print "------------------------------------"
			return None
		from sklearn.metrics import roc_curve, auc
		fpr, tpr, thresholds = roc_curve(pr, ac)
		roc_auc = auc(fpr, tpr)
		# build & display ROC plot (MUST HAVE '%matplotlib inline' if running in iPython)
		plt.clf()
		plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()	

	def single_custom_analysis(self, features):
		prepd_feat = self.scaler.transform(features)
		return self.clf.predict_proba(prepd_feat)
		#predict_scores = predict_proba(prepd_feat)
		#results = []
		#for each in predict_


# shuffle and split training and test sets
X, y = shuffle(X, y, random_state=random_state)
half = int(n_samples / 2)
X_train, X_test = X[:half], X[half:]
y_train, y_test = y[:half], y[half:]

# Run classifier
classifier = svm.SVC(kernel='linear', probability=True)
probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

