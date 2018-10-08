
# numpy implementation of groupby function 
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.keys = keys
        self.n_keys = max(self.keys_as_int)
        self.set_indices()

    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
      #print(self.indices[0], self.keys[self.indices[0]] )

    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result

#######################################################################
# Evaluate
def evaluate(qf,ql,qp,gf,gl,gp):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
#    print(np.reshape(gl, (gl.shape[0],1).shape))
    result = Groupby(gl).apply(np.mean, score, broadcast=False)
    inde = np.argsort(result)
    inde = inde[::-1]
    gl = np.unique(gl)
   # print(result.shape)
    if ql!=gl[inde[0]]:
        most_simi = (ql,gl[inde[0]] )
        s = score[inde[0]]
    else:
        most_simi = (ql,gl[inde[1]])
        s = score[inde[1]]
    return most_simi, s
