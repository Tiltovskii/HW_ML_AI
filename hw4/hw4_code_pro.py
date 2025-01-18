import numpy as np
from collections import Counter
from typing import Tuple, Callable


def calculate_information_gain(
        left_counts_of_ones: float, 
        left_position: float, 
        all_counts: float,
        max_position: float, 
        impurity_funcion: Callable
    ) -> float:
    
    R_l = left_position
    R_r = max_position - left_position
    R = max_position
    left_impurity = impurity_funcion(left_counts_of_ones/R_l)
    right_impurity = impurity_funcion((all_counts - left_counts_of_ones)/R_r)
    return R_l/R*left_impurity + R_r/R*right_impurity


def calculate_binary_gini(p: float) -> float:
    return 1 - p**2 - (1 - p)**2


def calculate_binary_entropy(p: float) -> float:
    return -p*np.log(p) - (1 - p)*np.log(1 - p)


def find_best_split(
        feature_vector: np.ndarray, 
        target_vector: np.ndarray,
        impurity_function: Callable[[float], float],
        min_samples_split: int | None = None
    ) -> Tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = \frac {|R_l|}{|R|}H(R_l) + \frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    legnth = len(feature_vector)
    sorted_indx = feature_vector.argsort()
    sorted_feature_vector, sorted_target_vector = feature_vector[sorted_indx], target_vector[sorted_indx]
    
    thresholds = (sorted_feature_vector[1:] + sorted_feature_vector[:-1]) / 2
    
    unique_thresholds_idx = np.array(list(map(lambda x: np.all(x != sorted_feature_vector), thresholds)))
    
    if min_samples_split is not None and np.sum(unique_thresholds_idx) <= 2*min_samples_split + 1:
        return None, None, None, None
    
    if min_samples_split is not None:
        thresholds = thresholds[unique_thresholds_idx][min_samples_split:-min_samples_split]
    else:
        thresholds = thresholds[unique_thresholds_idx]
        
    vcalculate_information_gain = np.vectorize(calculate_information_gain)
    information_gains = vcalculate_information_gain(
        left_counts_of_ones=np.cumsum(sorted_target_vector[:-1]), 
        left_position=np.arange(1, legnth),
        all_counts=np.sum(sorted_target_vector),
        max_position=legnth,
        impurity_funcion=impurity_function
    )
    
    if min_samples_split is not None:
        unique_information_gains = information_gains[unique_thresholds_idx][min_samples_split:-min_samples_split]
    else:
        unique_information_gains = information_gains[unique_thresholds_idx]
    
    indx_best = np.argmin(unique_information_gains)
    
    threshold_best, information_gains_best = thresholds[indx_best], unique_information_gains[indx_best]
    return thresholds, unique_information_gains, information_gains_best, threshold_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if (
            np.all(sub_y == sub_y[0]) or 
            (self._max_depth is not None and depth == self._max_depth) or 
            (self._min_samples_leaf is not None and len(sub_y) <= self._min_samples_leaf)
        ):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            
            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, gini, threshold = find_best_split(feature_vector, sub_y, calculate_binary_gini, self._min_samples_split)
            if gini is not None and (gini_best is None or gini > gini_best):
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        elem = x[node['feature_split']]
        feature_type = self._feature_types[node['feature_split']]
        if feature_type == 'real':
            if elem < node["threshold"]:
                return self._predict_node(x, node['left_child'])
            return self._predict_node(x, node['right_child'])
        
        if elem in node["categories_split"]:
            return self._predict_node(x, node['left_child'])
        return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **kwargs):
        self._max_depth = kwargs['max_depth'] 
        self._min_samples_split = kwargs['min_samples_split'] 
        self._min_samples_leaf = kwargs['min_samples_leaf'] 
