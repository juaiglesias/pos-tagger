import matplotlib.pyplot as plt
import pandas as pd

# Test
test_errors = pd.read_csv('test_errors.txt')

n_experiments, n_test_cases = test_errors.shape # 5,6 

x_names = list(test_errors.columns.values)
x_values = range(1,n_test_cases+1)

plt.figure(figsize=(8,5))

for experiment_number in range(n_experiments):

    exp = test_errors.iloc[experiment_number,:]
    lab = 'Experiment ' + str(experiment_number+1)

    plt.plot(x_values, exp, label=lab)

plt.ylim(0, 1)
plt.axhline(y=0.5, linestyle='--', color='k')
plt.xlabel('Test Case')
plt.ylabel('Error Rate')
plt.xticks(x_values, x_names)
plt.legend()
plt.tight_layout()
plt.savefig('test_errors.png')
    
# Ambiguous
ambiguous_errors = pd.read_csv('ambiguous_errors.txt')

n_experiments, n_test_cases = ambiguous_errors.shape # 5,6 

x_names = list(ambiguous_errors.columns.values)
x_values = range(1,n_test_cases+1)

plt.figure(figsize=(8,5))

for experiment_number in range(n_experiments):

    exp = ambiguous_errors.iloc[experiment_number,:]
    lab = 'Experiment ' + str(experiment_number+1)

    plt.plot(x_values, exp, label=lab)

plt.ylim(0, 1)
plt.axhline(y=0.5, linestyle='--', color='k')
plt.xlabel('Test Case')
plt.ylabel('Ambiguous Error Rate')
plt.xticks(x_values, x_names)
plt.legend()
plt.tight_layout()
plt.savefig('ambiguous_errors.png')

# OOV
oov_errors = pd.read_csv('oov_errors.txt')

n_experiments, n_test_cases = oov_errors.shape # 5,6 

x_names = list(oov_errors.columns.values)
x_values = range(1,n_test_cases+1)

plt.figure(figsize=(8,5))

for experiment_number in range(n_experiments):

    exp = oov_errors.iloc[experiment_number,:]
    lab = 'Experiment ' + str(experiment_number+1)

    plt.plot(x_values, exp, label=lab)

plt.ylim(0, 1)
plt.axhline(y=0.5, linestyle='--', color='k')
plt.xlabel('Test Case')
plt.ylabel('OOV Error Rate')
plt.xticks(x_values, x_names)
plt.legend()
plt.tight_layout()
plt.savefig('oov_errors.png')
