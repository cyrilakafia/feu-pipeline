from inf import run_inference

best_assigns, best_params = run_inference('/Users/csa46/Documents/Computation/functional encoding units/feu-with1231/sim1231_true.p', 
                                          title='demo', 
                                          device='cpu', 
                                          iterations=5, 
                                          concentration=1, 
                                          max_clusters=20, 
                                          timepoint=0, 
                                          seed=None
                                          )

print(type(best_assigns))
print(type(best_params))

# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.frame.DataFrame'>ex

