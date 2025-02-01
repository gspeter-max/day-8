import pandas 
from scipy.sparse import csr_matrix 
from implicit.als import AlternatingLeastSquares  

data = {
    "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    "video_id": [101, 102, 103, 101, 104, 102, 103, 105, 106, 107],
    "watch_count": [5, 3, 2, 4, 1, 7, 5, 2, 3, 4]
} 

df = pd.DataFrame(data) 

sparse_matrix = csr_matrix((df['watch_count'],(df['user_id'],df['video_id'])))

model = AlternatingLeastSquares(
    factors = 160, 
    iterations = 15, 
    regularization = 0.1
)

model.fit(sparse_matrix)
user_id = 1 
recommendations = model.recommend(user_id , sparse_matrix[user_id], N = 5 ) 
print(recommendations) 
