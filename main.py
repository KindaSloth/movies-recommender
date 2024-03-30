import os

from surprise import Dataset, Reader, KNNBaseline 

def read_items_names():
    file_name = "./datasets/movies.csv"
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, encoding="ISO-8859-1") as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    
    return rid_to_name, name_to_rid

file_path = os.path.expanduser("~/Dev/movies-recommender/datasets/ratings.csv")
reader = Reader(line_format="user item rating", sep=",")

data = Dataset.load_from_file(file_path, reader=reader)

trainset = data.build_full_trainset()
sim_options = {"name": "cosine", "user_based": False}
algo = KNNBaseline(sim_options=sim_options)
algo.fit(trainset)

rid_to_name, name_to_rid = read_items_names()

for movie in name_to_rid:
    print(movie)

print()
movie_name = input("Which movie from the list you like? ")

raw_id = name_to_rid[movie_name]
inner_id = algo.trainset.to_inner_iid(raw_id)

story_neighbors = algo.get_neighbors(inner_id, k=1)
story_neighbors = (
    algo.trainset.to_raw_iid(inner_id) for inner_id in story_neighbors
)
story_neighbors = (rid_to_name[rid] for rid in story_neighbors)

print()
print("The nearest neighbors are:")
for movie in story_neighbors:
    print(movie)
