from inf import run_inference
from pandas import DataFrame

best_assigns, best_params = run_inference('../feu-with1231/sim1234_true.p', title='demo', device='cpu', iterations=5, seed=None)

print(type(best_assigns))
print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>

