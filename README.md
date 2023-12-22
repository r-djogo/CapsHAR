# Fresnel Zone-Based Voting with Capsule Networks for Human Activity Recognition From Channel State Information

Code for paper: "Fresnel Zone-Based Voting with Capsule Networks for Human Activity Recognition From Channel State Information", submitted to the Internet of Things Journal (IoTJ).

## Repo Structure
1. fresnel_voting - contains code for fresnel voting experiments
2. hm_dataset - contains code for experiments on HM interaction dataset
3. signfi - contains code for experiments on SignFi dataset
4. widar3 - contains code for experiments on Widar3.0 dataset
5. yousefi_etal - contains code for experiments on Yousefi et al. dataset

## Notes

We have also developped a real-time demonstration of Wi-Fi-based human activity recognition for the 2023 IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC 2023). For a video showing a demonstration of a live implementation for the CapsHAR model, see our other GitHub (https://github.com/r-djogo/har_demo).

GPU acceleration was used for training/testing: <br>
Cuda compilation tools, release 11.6, V11.6.124 <br>
With a NVIDIA TITAN Xp GPU

Code based partially on implementations from: 
- https://github.com/EscVM/Efficient-CapsNet
- https://github.com/ermongroup/Wifi_Activity_Recognition
- https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset

## Data
- Link to HM dataset from paper: https://github.com/salehinejad/CSI-joint-activ-orient (once made available)
- Link to Widar3.0 dataset from paper: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
- Link to Yousefi et al. dataset from paper: https://github.com/ermongroup/Wifi_Activity_Recognition
- Link to SignFi dataset from paper: https://yongsen.github.io/SignFi/
