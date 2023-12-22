from HAR_CSI import *
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, accuracy_score

def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label

predictions = np.load('./file.npy', allow_pickle=True).item()

APs = ['AP0','AP1','AP2','AP3','AP4','AP5'] # APs to be included

runs = list(range(10))

##### VOTING - equal weights
weights = [np.sqrt([1.,1.,1.,1.,1.,1.]),
           np.sqrt([1.,1.,1.,1.,1.,1.]),
           np.sqrt([1.,1.,1.,1.,1.,1.]),
           np.sqrt([1.,1.,1.,1.,1.,1.]),
           np.sqrt([1.,1.,1.,1.,1.,1.])]
B_ACC = []

for run in runs:
    # voting
    y_true = onehot_encoding(predictions['label'+str(run)][:,0], 10)
    loc = predictions['label'+str(run)][:,1]
    ori = predictions['label'+str(run)][:,2]
    
    voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
    for ap_id, ap_name in enumerate(APs):
        voted_pred += [[weights[l-1][ap_id]]*10 for l in loc] * predictions[str(run)+ap_name]
    B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

# print("Balanced accuracy:", B_ACC)
print("EQUAL WEIGHTS avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - FSPL
left = -2.5
right = 0
top = 0
bottom = -2.5

tx_x = right-0.5
tx_y = top-0.5

user_x = [tx_x-1.365, tx_x-0.455, tx_x-0.455, tx_x-1.365, tx_x-0.91]
user_y = [tx_y-0.455, tx_y-0.455, tx_y-1.365, tx_y-1.365, tx_y-0.91]

ap_x = [tx_x-0.5, tx_x-1.4, tx_x-2, 0, 0, tx_x]
ap_y = [0, 0, tx_y, tx_y-0.5, tx_y-1.4, tx_y-2]

fr = 5825.0*1e6
c = 2.997925e8
wavelen = c/fr

B_ACC = []

weights = np.zeros([5,6])
for ap in range(6):
    for loc in range(5):
        distance = np.sqrt((tx_x - user_x[loc])**2 + (tx_y - user_y[loc])**2) + \
                np.sqrt((ap_x[ap] - user_x[loc])**2 + (ap_y[ap] - user_y[loc])**2)
        weights[loc][ap] = 1/distance**2 # free space pathloss proportional to: 1/dist^2

for run in runs:
    # voting
    y_true = onehot_encoding(predictions['label'+str(run)][:,0], 10)
    loc = predictions['label'+str(run)][:,1]
    ori = predictions['label'+str(run)][:,2]
    
    voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
    for ap_id, ap_name in enumerate(APs):
        voted_pred += [[weights[l-1][ap_id]]*10 for l in loc] * predictions[str(run)+ap_name]
    B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

# print("Balanced accuracy:", B_ACC)
print("FSPL WEIGHTS avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC), '\n')


##### VOTING - location plus orientation
left = -2.5
right = 0
top = 0
bottom = -2.5

tx_x = right-0.5
tx_y = top-0.5

user_x = [tx_x-1.365, tx_x-0.455, tx_x-0.455, tx_x-1.365, tx_x-0.91]
user_y = [tx_y-0.455, tx_y-0.455, tx_y-1.365, tx_y-1.365, tx_y-0.91]

ap_x = [tx_x-0.5, tx_x-1.4, tx_x-2, 0, 0, tx_x]
ap_y = [0, 0, tx_y, tx_y-0.5, tx_y-1.4, tx_y-2]

fr = 5825.0*1e6
c = 2.997925e8
wavelen = c/fr

normal_dir_loc_ap = np.zeros([5,6])
n = [[79-1, 36-1, 9-1, 101-1, 80-1, 59-1],
     [35-1, 20-1, 10-1, 35-1, 20-1, 10-1],
     [101-1, 80-1, 59-1, 79-1, 36-1, 9-1],
     [128-1, 90-1, 56-1, 128-1, 90-1, 56-1],
     [80-1, 51-1, 28-1, 80-1, 51-1, 28-1]]

for loc in range(5):
    for ap in range(len(APs)):
        x1 = tx_x
        x2 = ap_x[ap]
        y1 = tx_y
        y2 = ap_y[ap]
        
        d = 1/2*np.sqrt((x2-x1)**2+(y2-y1)**2)
        a = d + n[loc][ap]*wavelen/4
        r = np.sqrt((d+n[loc][ap]*wavelen/4)**2 - d**2)

        # find angle to closest point on surface of ellipse to user within margin of error
        accuracy = 1e-8
        w = np.arctan2(y2-y1,x2-x1)
        t = np.linspace(0,2*np.pi,num=300)
        dist2 = (user_x[loc] - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
                + (user_y[loc] - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
        sort_dist2 = np.sort(dist2)
        ind = np.argmin(dist2)
        best_t = t[ind]
        
        while (sort_dist2[2] - sort_dist2[1]) > accuracy:
            t = np.linspace(t[ind-1], t[ind+1], num=99)
            dist2 = (user_x[loc] - ((x1+x2)/2 + a*np.cos(t)*np.cos(w) - r*np.sin(t)*np.sin(w)))**2 \
                    + (user_y[loc] - ((y1+y2)/2 + a*np.cos(t)*np.sin(w) + r*np.sin(t)*np.cos(w)))**2
            sort_dist2 = np.sort(dist2)
            ind = np.argmin(dist2)
            best_t = t[ind]

        X = a*np.cos(best_t)
        Y = r*np.sin(best_t)
        closest_x = (x1+x2)/2 + X*np.cos(w) - Y*np.sin(w)
        closest_y = (y1+y2)/2 + X*np.sin(w) + Y*np.cos(w)

        normal_dir_loc_ap[loc,ap] = np.arctan2((closest_y-user_y[loc]), (closest_x-user_x[loc]))

# % loc : ap0, ap1, ap2, ap3, ap4, ap5 
# % 1: 79, 36, 9, 101, 80, 59
# % 2: 35, 20, 10, 35, 20, 10
# % 3: 101, 80, 59, 79, 36, 9
# % 4: 128, 90, 56, 128, 90, 56
# % 5: 80, 51, 28, 80, 51, 28

# weights = [np.sqrt([1./79,1./36,1./9,1./101,1./80,1./59]),
#            np.sqrt([1./35,1./20,1./10,1./35,1./20,1./10]),
#            np.sqrt([1./101,1./80,1./59,1./79,1./36,1./9]),
#            np.sqrt([1./128,1./90,1./56,1./128,1./90,1./56]),
#            np.sqrt([1./80,1./51,1./28,1./80,1./51,1./28])]
orients = np.pi/180 * np.array([135.0, 90.0, 45.0, 0.0, -45.0])

betas = np.linspace(0.0, 1.0, num=41)
best_acc = 0
best_std = 0
best_beta = 0
beta_accs = []
for beta in betas:
    B_ACC = []

    for run in runs:
        # voting
        y_true = onehot_encoding(predictions['label'+str(run)][:,0], 10)
        loc = predictions['label'+str(run)][:,1]
        ori = predictions['label'+str(run)][:,2] - 1
        true_orients = np.array([[orients[int(o)]] for o in ori])

        voted_pred = np.zeros((predictions[str(run)+'AP0'].shape[0], predictions[str(run)+'AP0'].shape[1]))
        for ap_id, ap_name in enumerate(APs):
            # print(np.array([[normal_dir_loc_ap[l-1,ap_id]] for l in loc]).shape,
            #       np.array([[weights[l-1][ap_id]] for l in loc]).shape,
            #       predictions[str(run)+ap_name].shape)
            # exit()
            voted_pred += (1.0 + beta*np.cos(true_orients - (np.array([[normal_dir_loc_ap[l-1,ap_id]] for l in loc])))) \
                        * np.array([[weights[l-1][ap_id]] for l in loc]) * predictions[str(run)+ap_name]
        B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(voted_pred, axis=1)))

    if np.mean(B_ACC) > best_acc:
        best_acc = np.mean(B_ACC)
        best_std = np.std(B_ACC)
        best_beta = beta

    beta_accs.append(np.mean(B_ACC))

# print(list(betas))
# print(beta_accs)
print(best_beta)
# print("Balanced accuracy:", B_ACC)
print("LOC+ORI WEIGHTS avg balanced accuracy:", best_acc, "std balanced accuracy:", best_std, '\n')


# ##### VOTING - grid search

# best_B_ACC = 0
# best_B_ACC_std = 0
# best_weights = [0,0,0,0,0,0]
# t_start = time.time()

# w = [list(np.linspace(0,1,11)) for i in range(5)]
# w.insert(0,[1.0])

# W = list(itertools.product(*w))
# pbar = ProgressBar(widgets=[Percentage(), Bar(), Timer()], maxval=len(W)).start()
# i = 0

# for weights in itertools.product(*w):
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
# print(best_B_ACC, best_B_ACC_std, best_weights)

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
    for run in runs:
        y_true = onehot_encoding(predictions['label'+str(run)][:,0], 10)
        B_ACC.append(balanced_accuracy_score(np.argmax(y_true, axis=1), np.argmax(predictions[str(run)+ap_name], axis=1)))
    
    print(ap_name, "avg balanced accuracy:", np.mean(B_ACC), "std balanced accuracy:", np.std(B_ACC))