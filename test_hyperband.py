from snorkel.learning import GenerativeModelWeights
from snorkel.learning.structure import generate_label_matrix

weights = GenerativeModelWeights(10)
for i in range(10):
    weights.lf_accuracy[i] = 2.5
weights.dep_similar[0, 1] = 0.25
weights.dep_similar[2, 3] = 0.25

L_gold_train, L_train = generate_label_matrix(weights, 10000)
L_gold_dev, L_dev = generate_label_matrix(weights, 1000)


from snorkel.learning import GenerativeModel
from snorkel.learning import HyperbandSearch

param_ranges = {
    'step_size' : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'decay' : [1.0, 0.95, 0.9],
}

searcher = HyperbandSearch(GenerativeModel, param_ranges, 30, L_train)

gen_model, run_stats = searcher.fit(L_dev, L_gold_dev)

print(run_stats)
