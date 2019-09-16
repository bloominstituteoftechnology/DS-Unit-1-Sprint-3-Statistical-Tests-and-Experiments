#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'module1-statistics-probability-and-inference'))
	# print(os.getcwd())
except:
	pass
#%% [markdown]
# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# <br></br>
# 
# ## *Data Science Unit 1 Sprint 3 Assignment 1*
# 
# # Apply the t-test to real data
# 
# Your assignment is to determine which issues have "statistically significant" differences between political parties in this [1980s congressional voting data](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records). The data consists of 435 instances (one for each congressperson), a class (democrat or republican), and 16 binary attributes (yes or no for voting for or against certain issues). Be aware - there are missing values!
# 
# Your goals:
# 
# 1. Load and clean the data (or determine the best method to drop observations when running tests)
# 2. Using hypothesis testing, find an issue that democrats support more than republicans with p < 0.01
# 3. Using hypothesis testing, find an issue that republicans support more than democrats with p < 0.01
# 4. Using hypothesis testing, find an issue where the difference between republicans and democrats has p > 0.1 (i.e. there may not be much of a difference)
# 
# Note that this data will involve *2 sample* t-tests, because you're comparing averages across two groups (republicans and democrats) rather than a single group against a null hypothesis.
# 
# Stretch goals:
# 
# 1. Refactor your code into functions so it's easy to rerun with arbitrary variables
# 2. Apply hypothesis testing to your personal project data (for the purposes of this notebook you can type a summary of the hypothesis you formed and tested)

#%%
### YOUR CODE STARTS HERE
import pandas
import numpy

cols = ['party','handicapped-infants','water-project',
		'budget','physician-fee-freeze', 'el-salvador-aid',
		'religious-groups','anti-satellite-ban',
		'aid-to-contras','mx-missile','immigration',
		'synfuels', 'education', 'right-to-sue','crime','duty-free',
		'south-africa']
get_ipython().system('wget -N https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')


df = pandas.read_csv('house-votes-84.data',
               header=None,
               names=cols).replace({'?':numpy.NaN, 'n':0, 'y':1})
df.head()

#%%
import scipy.stats

democrats = df[df['party']=='democrat']
republicans = df[df['party']=='republican']
for column in cols[1:]:
	result = scipy.stats.ttest_ind(republicans[column], democrats[column], nan_policy='omit')
	demPercent = democrats[column].mean()*100
	repPercent = republicans[column].mean()*100
	if result[1] < 0.05:
		print(f'Due to a p-value of {result[1]:.6} < 0.05, we reject the null hypothesis that democrat ({demPercent:.4}% in favor) and republican ({repPercent:.4}% in favor) house members vote similarly on {column}.')
	else:
		print(f'Due to a p-value of {result[1]:6.6} > 0.05, we fail to reject the null hypothesis that democrat and republican house members vote similarly on {column}.')

#%%
# Extra 1-sample t-testing added during lecture

result = scipy.stats.ttest_1samp(democrats['synfuels'], 0.5, nan_policy='omit')
demPercent = democrats['synfuels'].mean()*100
print(f'Due to a p-value of {result[1]:.6} > 0.05, we fail to reject the null hypothesis that democrat house members ({demPercent:.4}% in favor) do not vote significantly in favor or against synfuels.')
result = scipy.stats.ttest_1samp(republicans['synfuels'], 0.5, nan_policy='omit')
repPercent = republicans['synfuels'].mean()*100
print(f'Due to a p-value of {result[1]:.6} < 0.05, we reject the null hypothesis that republican house members ({repPercent:.4}% in favor) do not vote significantly in favor or against synfuels.')


#%%
