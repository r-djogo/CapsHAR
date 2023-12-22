from HAR_CSI import *
import time
import itertools
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score

predictions = np.load('./path_to_file/AP_predictions.npy', allow_pickle=True).item()
APs = ['AP0','AP1','AP2','AP3','AP4'] # APs to be included
runs = 10


##### VOTING - even
B_ACC = []
weights = np.sqrt([1.,1.,1.,1.,1.])

for run in range(runs):
    # voting
    y_true = predictions['label'+str(run)]
    voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
    for ap_id, ap_name in enumerate(APs):
        voted_pred += weights[ap_id] * predictions[str(run)+ap_name]
    B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

# print("Balanced accuracy:", B_ACC)
print("EVEN WEIGHTS avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - FSPL
left = 0
right = 5.6
top = 6.05
bottom = 0

user_x = right-2.9
user_y = top-3.8

tx_x = right-2.85
tx_y = top-3.15

ap_x = [right-2.9, 0.78, 0.7, 0.95, right-2.9]
ap_y = [top-0.51, top-1.09, 0.56+2, 0.56, 0.31]

fr = 2401*1e6
c = 2.997925e8
wavelen = c/fr

B_ACC = []

weights = np.zeros(5)
for ap in range(5):
    distance = np.sqrt((tx_x - user_x)**2 + (tx_y - user_y)**2) + \
               np.sqrt((ap_x[ap] - user_x)**2 + (ap_y[ap] - user_y)**2)
    weights[ap] = 1/distance**2 # free space pathloss proportional to: 1/dist^2

for run in range(runs):
    # voting
    y_true = predictions['label'+str(run)]
    voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
    for ap_id, ap_name in enumerate(APs):
        voted_pred += weights[ap_id] * predictions[str(run)+ap_name]
    B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

# print("Balanced accuracy:", B_ACC)
print(weights)
print("FSPL WEIGHTS avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - location plus orientation
left = 0
right = 5.6
top = 6.05
bottom = 0

user_x = right-2.9
user_y = top-3.8

tx_x = right-2.85
tx_y = top-3.15

ap_x = [right-2.9, 0.78, 0.7, 0.95, right-2.9]
ap_y = [top-0.51, top-1.09, 0.56+2, 0.56, 0.31]

fr = 2401*1e6
c = 2.997925e8
wavelen = c/fr

normal_dir_ap = [0, 0, 0, 0, 0]
n = [21-1, 18-1, 10-1, 3-1, 1]

for i in range(len(APs)):
    x1 = tx_x
    x2 = ap_x[i]
    y1 = tx_y
    y2 = ap_y[i]
    
    d = 1/2*np.sqrt((x2-x1)**2+(y2-y1)**2)
    a = d + n[i]*wavelen/4
    r = np.sqrt((d+n[i]*wavelen/4)**2 - d**2)
    
    # find angle to closest point on surface of ellipse to user within margin of error
    accuracy = 1e-8
    w = np.arctan2(y2-y1,x2-x1)
    t = np.linspace(0,2*np.pi,num=300)
    dist2 = (user_x - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
            + (user_y - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
    sort_dist2 = np.sort(dist2)
    ind = np.argmin(dist2)
    best_t = t[ind]
    
    while (sort_dist2[2] - sort_dist2[1]) > accuracy:
        t = np.linspace(t[ind-1], t[ind+1], num=99)
        dist2 = (user_x - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
                + (user_y - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
        sort_dist2 = np.sort(dist2)
        ind = np.argmin(dist2)
        best_t = t[ind]

    X = a*np.cos(best_t)
    Y = r*np.sin(best_t)
    closest_x = (x1+x2)/2 + X*np.cos(w) - Y*np.sin(w)
    closest_y = (y1+y2)/2 + X*np.sin(w) + Y*np.cos(w)

    normal_dir_ap[i] = np.arctan2((closest_y-user_y), (closest_x-user_x))

orients = np.pi/180 * np.array([0.0, 45.0, 90.0, 180.0]) + np.pi/2 #90 degress added to align axes

betas = np.linspace(0.0, 0.85, num=35)
best_acc = 0
best_std = 0
best_beta = 0
beta_accs = []
for beta in betas:
    B_ACC = []

    for run in range(runs):
        # voting
        y_true = predictions['label'+str(run)]
        y_orient = predictions['orient'+str(run)]
        true_orients = np.array([[orients[int(ori)]] for ori in y_orient])

        voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
        for ap_id, ap_name in enumerate(APs):
            if ap_id == 4:
                voted_pred += weights[ap_id] * predictions[str(run)+ap_name]
            else:
                voted_pred += (1.0 + beta*np.cos(true_orients - (np.ones_like(true_orients) * normal_dir_ap[i]))) \
                            * weights[ap_id] * predictions[str(run)+ap_name]
        B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

    if np.mean(B_ACC) > best_acc:
        best_acc = np.mean(B_ACC)
        best_std = np.std(B_ACC)
        best_beta = beta

    beta_accs.append(np.mean(B_ACC))

# print(list(betas))
# print(beta_accs)
print(best_beta)
print("LOC+ORI WEIGHTS avg balanced accuracy:", best_acc, "std balanced accuracy:", best_std, '\n')


# ##### VOTING - grid search
# best_B_ACC = 0
# best_B_ACC_std = 0
# best_weights = [0,0,0,0,0]
# t_start = time.time()

# w = [list(np.linspace(0,1,41)) for i in range(4)]
# w.append([1.0])

# W = list(itertools.product(*w))
# pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=len(W)).start()
# i = 0

# for weights in itertools.product(*w):
#     # skip scenarios that are bad
#     if weights[0] >= weights[1] and weights[1] >= weights[2] and weights[3] >= weights[0] and weights[4] >= weights[3]:
#         pass
#     else:
#         i +=1
#         pbar.update(i)
#         continue

#     matrices = []
#     MCC = []
#     F1 = []
#     B_ACC = []
#     ACC = []

#     for run in range(10):
#         # voting
#         y_true = predictions['label'+str(run)]
#         voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
#         for ap_id, ap_name in enumerate(APs):
#             voted_pred += weights[ap_id] * predictions[str(run)+ap_name]
#         B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

#     if np.mean(B_ACC) > best_B_ACC:
#         best_B_ACC = np.mean(B_ACC)
#         best_B_ACC_std = np.std(B_ACC)
#         best_weights = weights

#     i +=1
#     pbar.update(i)

# pbar.finish()
# print('Time = ' ,time.time() - t_start)
# print('GRID SEARCH WEIGHTS', best_B_ACC, best_B_ACC_std, best_weights, '\n')

# # print("Avg model matrix:\n")
# # print(avg_matrix)
# # print("Std model matrix:\n")
# # print(std_matrix)
# # print("Balanced accuracy:")
# # print(B_ACC)
# # print("Avg balanced accuracy:")
# # print(np.mean(B_ACC))
# # print("Std balanced accuracy:")
# # print(np.std(B_ACC))


##### individual APs

for ap_id, ap_name in enumerate(APs):
    B_ACC = []
    for run in range(runs):
        y_true = predictions['label'+str(run)]
        B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(predictions[str(run)+ap_name], axis=1)))

    print(ap_name, "avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC))
