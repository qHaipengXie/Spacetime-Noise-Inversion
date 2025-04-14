import pickle


p = 0.01
n = range(20)
res = [1-p*4**(i+1)/(4**(i+1)-1) for i in n]
with open('./Plot/Raw','wb')as f:
    pickle.dump(res,f)



