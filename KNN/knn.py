import numpy as np

# create the class
class KNearestNeighbor:
    # create the init method
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # get the distance matrix
        distances = self.get_distance_matrix(X_test)
        return self.predict_labels(distances)

    # create the distance matrix
    def get_distance_matrix(self, X_test: np.ndarray) -> np.ndarray:
        # get the dimensions of the distance matrix
        num_row = X_test.shape[0]
        num_col = self.X_train.shape[0]

        # create the distance matrix
        distances = np.zeros((num_row, num_col))

        # fill in the entries in the matrix
        # for each point in the test dataset, we are storing the
        # euclidean distance against each training examples.
        for i in range(num_row):
            for j in range(num_col):
                distances[i,j] = np.sqrt(np.sum(((X_test[i,:] - self.X_train[j,:]))**2))
        
        return distances
    
    def predict_labels(self, distances: np.ndarray):
        num_entries = distances.shape[0]
        y_pred = np.zeros(num_entries)

        for i in range(num_entries):
            y_index = np.argsort(distances[i,:])
            # grab the first k elements
            first_k_elt = self.y_train[y_index][:self.k].astype(int)

            # perform majority voting
            y_pred[i] = np.argmax(np.bincount(first_k_elt))
        return y_pred


if __name__=="__main__":
    # get the data
    X = np.loadtxt("data.txt", delimiter=",")
    y = np.loadtxt("targets.txt")

    KNN = KNearestNeighbor(k=3)
    KNN.fit(X,y)
    y_predicted = KNN.predict(X)
    print(f"The accuracy is: {np.mean(y == y_predicted) * 100:.3f}%")