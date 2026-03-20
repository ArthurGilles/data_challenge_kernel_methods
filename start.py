import time
import numpy as np 
import pandas as pd 
from scipy.optimize import minimize


class binary_SVM_classifier:
    """
    I used the binary "C-SVM" classifier from slide 161 of the course.
    """
    def __init__(self, kernel, C=1.0):
        # kernel should be a function as defined below for the rbf kernel
        self.kernel = kernel
        self.C = C

    def get_gram_matrix(self, X):
        return self.kernel(X,X) 

    def fit(self, X, y):
        # y should contain 1 or -1
        n = X.shape[0]
        K = self.get_gram_matrix(X)
        alpha0 = np.zeros(n) # inital values of alpha
        
        def objective(alpha):
            # I used the optimization problem formulation from slide 161
            # I put a minus sign because I use scipy minimizes functions (and the formulation uses a max)
            return (alpha.T @ K @ alpha) - (2 * alpha.T @ y)
        
        # I also defined a jacobian function to speed up the optimization
        def jacobian(alpha):
            return (2 * K @ alpha) - (2 * y)

        # Constraints
        bounds = []
        for i in range(n):
            if y[i] > 0:
                bounds.append((0.0, self.C))
            else:
                bounds.append((-self.C, 0.0))
        
        # Optimization
        result = minimize(
            fun=objective, 
            x0=alpha0, 
            jac=jacobian, 
            bounds=bounds, 
            method="L-BFGS-B" # Did not really know which method to use here...
        )

        self.alpha = result.x
        
        # Support vectors
        self.alpha_sv = self.alpha[np.abs(self.alpha) > 1e-5]
        self.support_vectors = X[np.abs(self.alpha) > 1e-5]

        return self.alpha
        
    def predict(self, X_new):
        # Only use support vector because the solution is usually sparse
        K_vals = self.kernel(X_new, self.support_vectors)
        y_pred = K_vals @ self.alpha_sv
        
        return y_pred
    


class SVM_classifier:
    """
    Now I have defined a binary SVM classifier, I define a multi-class SVM
    classifier, by using a one-to-one method.
    If there are n classes, I create a SVM for every pair of classes, and to
    predict a sample, I make all the classifiers vote, and I predict the class which
    gets the most votes.
    """

    def __init__(self, kernel, C=1.0):
        self.kernel = kernel
        self.C = C
    
    def fit(self, X, y):
        # I train one binary SVM for every pair of classes
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        self.classifiers = [[] for _ in range(self.n_classes - 1)]

        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
                idx = np.where((y == self.classes[i]) | (y == self.classes[j]))[0]
                X_subset = X[idx]
                y_subset = y[idx]
                
                y_binary = np.where(y_subset == self.classes[i], 1, -1)
                
                classifier = binary_SVM_classifier(kernel=self.kernel, C=self.C)
                classifier.fit(X_subset, y_binary)
                
                self.classifiers[i].append(classifier)
    
    def predict(self, X_new):
        # I make all the trained binary SVMs vote
        # If there is a tie, I use the sum of the predictions to get
        # which class votes are more confident
        n_samples = X_new.shape[0]
        
        votes = np.zeros((n_samples, self.n_classes))
        confidences = np.zeros((n_samples, self.n_classes))
        
        for i in range(self.n_classes - 1):
            for k, j in enumerate(range(i + 1, self.n_classes)):
                clf = self.classifiers[i][k]
                
                preds = clf.predict(X_new)
                
                votes[:, i] += (preds > 0).astype(int)
                votes[:, j] += (preds <= 0).astype(int)
                
                confidences[:, i] += preds
                confidences[:, j] -= preds 
                
        predicted_indices = np.zeros(n_samples, dtype=int)
        
        for s in range(n_samples):
            max_vote = np.max(votes[s])
            candidates = np.where(votes[s] == max_vote)[0] 
            
            if len(candidates) == 1: # if no ties
                predicted_indices[s] = candidates[0]
            else: # if ties, use the sum of predictions
                best_candidate = candidates[np.argmax(confidences[s, candidates])]
                predicted_indices[s] = best_candidate
                
        return self.classes[predicted_indices]
    

def get_rbf_kernel(gamma=1.0):
    def kernel(X, Y):
        distance_sq = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * distance_sq)
    return kernel


def extract_features(X):
    N = X.shape[0]
    
    # Comment on the data:
    # We were told we were dealing with images. Looking at the .csv files,
    # We see there are 3072 data. We therefore understand that we are dealing
    # with 32*32*3 images (that we have to classify in 10 classes, so I think 
    # they are from CIFAR-10). I had to test how to reshape them to recover 
    # the original image (I tested (32,32,3), (32,3,32) and (3,32,32)), and
    # I saw that reshaping them to (3,32,32) recovered the images (even though
    # the colours were weird because of the normalization). I am using this 
    # information to extract spatial features from the data, because an SVM
    # on the raw pixels does not understand the spatial structure of the data.
    # I thus create features using spatial pooling and HOG

    images = X.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    
    features_list = []
    
    # SPATIAL POOLING
    # I cut the image in a 4*4 grid of 8*8 pixels and calculate mean and var
    blocks = images.reshape(N, 4, 8, 4, 8, 3)
    
    features_list.append(blocks.mean(axis=(2, 4)).reshape(N, -1))
    features_list.append(blocks.var(axis=(2, 4)).reshape(N, -1))
    
    # SIMPLIFIED HOG
    gray = images.mean(axis=3)
    
    # Calculate gradients
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    gy[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]

    # Calculate magnitude of gradient and put it in a 4*4 grid
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_blocks = magnitude.reshape(N, 4, 8, 4, 8)

    # Calculate angle of gradient, put it in bins of (180/n_bins) degrees and put it in a 4*4 grid
    n_bins = 8
    angle = np.degrees(np.arctan2(gy, gx)) % 180  # 0 to 180 degrees
    angle_bins = (angle // (180 / n_bins)).astype(int) % n_bins
    bin_blocks = angle_bins.reshape(N, 4, 8, 4, 8)
    
    # Create a 4*4 grid, and count the number of of gradients in this bin, pondered by their magnitude
    hog_hists = np.zeros((N, 4, 4, n_bins))
    
    for b in range(n_bins):
        mask_bin = (bin_blocks == b)
        
        pondered_number_in_bin = mask_bin * mag_blocks
        hog_hists[:, :, :, b] = pondered_number_in_bin.sum(axis=(2, 4))
        
    hog_features = hog_hists.reshape(N, -1)
    hog_features = hog_features / (np.linalg.norm(hog_features, axis=1).reshape(N, 1) + 1e-5)

    features_list.append(hog_features)
    
    return np.hstack(features_list)

def train_test_split(X, y, test_size=0.2):
    # train_test_split function like in sklearn
    split = int(len(X) * (1 - test_size))
    indices = np.arange(len(X))
    np.random.shuffle(indices)
        
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]



# Load, train and test code:

Xtr = np.array(pd.read_csv('./data/Xtr.csv',header=None,sep=',',usecols=range(3072))) 
Xte = np.array(pd.read_csv('./data/Xte.csv',header=None,sep=',',usecols=range(3072))) 
Ytr = np.array(pd.read_csv('./data/Ytr.csv',sep=',',usecols=[1])).squeeze() 

Xtr_feat = extract_features(Xtr)
Xte_feat = extract_features(Xte)

np.random.seed(42)
X_train, X_val, y_train, y_val = train_test_split(Xtr_feat, Ytr, test_size=0.2)

# This part of the code implements a grid search. When I was testing the algorithm
# to find the best parameters, I put a huge grid of values. Now that I have found the
# best parameters, I only have a 1*1 grid so that the code is much faster to test.

# Best values:
C_values = [10] 
gamma_values = [3.5]

best_acc = 0.0
best_C = None
best_gamma = None

# Train loop
print("Starting")
for C in C_values:
    for gamma in gamma_values: # grid search
        start_time = time.time()
        print(f"C={C}, gamma={gamma}")
        
        kernel = get_rbf_kernel(gamma=gamma)
        clf = SVM_classifier(kernel=kernel, C=C)
        
        clf.fit(X_train, y_train)
        
        y_pred_val = clf.predict(X_val)
        acc = np.mean(y_pred_val == y_val)
        
        elapsed_time = time.time() - start_time
        print(f"Accuracy: {acc:.4f} (Temps: {elapsed_time:.2f}s)")
        
        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_gamma = gamma

print(f"Best : C={best_C}, gamma={best_gamma}")
print(f"Best accuracy : {best_acc:.4f}\n")

best_kernel = get_rbf_kernel(gamma=best_gamma)
final_clf = SVM_classifier(kernel=best_kernel, C=best_C)

# Training on all the data, predict on Xte and save in Yte.csv
final_clf.fit(Xtr_feat, Ytr) 

y_te_pred = final_clf.predict(Xte_feat)

submission = pd.DataFrame({
    "Id": np.arange(1,len(y_te_pred)+1),
    "Prediction": y_te_pred.astype(int)
})

submission.to_csv("./data/Yte.csv", index=False)

