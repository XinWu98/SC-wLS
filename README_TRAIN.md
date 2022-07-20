## Training

#### Weight initialization and end-to-end optimization

```bash
python train_e2e_DLT.py <data_path> --dataset <dataset_name> --scene <scene_name>  -b1 --epochs 1500 --config_file configs/single_frame.ini --glr 1e-4 --model DenseSC_GNN --loss_classif 0.5 --loss_essential 10.0 --obj_geod_th 1.0 --loss_essential_init_epo 0 -Ea 5 -Eb 1e-4 <--cal_training_pose> <--useRANSAC> <--log_val_loss> --pretrained-model <SCNet_model_path>
```
Use `--e2e_training` to specify whether to train the weight network only, or do end-to-end training.

`--loss_classif` and `--loss_essential` are weights used to balance the classification loss and the regression loss. `-Ea` and `-Eb` are the $\alpha$ and $\beta$ in Eq. 11. As illustrated in the supplement, one should select proper hyper-parameters to balance the losses. We notice that for scenes with more noisy predicted coordinates (such as greatcourt), a higher weight of classification loss `--loss_classif` is preferred, which effectively prunes outliers. 

Besides, in Eq. 11, `-Ea` governs whether we focus on the first term (reducing pose errors) or the second one (avoiding the non-null space). `-Eb` is set to balance the trace value in the second term, whose magnitude could significantly vary across the training samples. Please refer to Section 4.1 of [1] for detailed theory analysis. One could adjust these hyper-parameters when training to see their influences.

To reproduce the results in the paper, one could use following settings:

##### Weight initialization:

We list the hyperparameters used in the command line. As mentioned above, the two terms in Eq. 11 play different roles, so at the early stage of weight initialization (index <sub>1</sub>) we focus more on the stablity, and later finetune the model (index <sub>2</sub>) with smaller leraning rate `--glr` and `--Ea` to enhance the accuracy when the former has converged. 

|     Scene     | loss_cla<sub>1</sub> | loss_ess<sub>1</sub> | Ea<sub>1</sub> | Eb<sub>1</sub> | glr<sub>1</sub> | loss_cla<sub>2</sub> | loss_ess<sub>2</sub> | Ea<sub>2</sub> | Eb<sub>2</sub> | glr<sub>2</sub> |
| :-----------: | :------------------: | :------------------: | :------------: | :------------: | :-------------: | :------------------: | :------------------: | :------------: | :------------: | :-------------: |
|     chess     |         0.5          |          10          |       5        |      1e-4      |      1e-4       |         0.5          |          10          |      3.5       |      1e-4      |      1e-5       |
|     heads     |         0.5          |          10          |       5        |      1e-4      |      1e-4       |          2           |          10          |       2        |      1e-4      |      1e-5       |
|     fire      |         0.5          |          10          |       5        |      1e-4      |      1e-4       |         0.5          |          10          |       1        |      1e-4      |      1e-5       |
|    office     |         0.5          |          10          |       5        |      1e-4      |      1e-4       |                      |                      |                |                |                 |
|    pumpkin    |         0.5          |          10          |       5        |      1e-4      |      1e-4       |          2           |          10          |       2        |      1e-4      |      1e-5       |
|    kitchen    |         0.5          |          10          |       5        |      1e-4      |      1e-4       |                      |                      |                |                |                 |
|    stairs     |         0.5          |          10          |       5        |      1e-4      |      1e-4       |         0.5          |          10          |       3        |      1e-4      |      1e-5       |
|  greatcourt   |          8           |          10          |       5        |      1e-6      |      1e-4       |          8           |          10          |       2        |      1e-7      |      1e-4       |
| kingscollege  |          2           |          10          |       5        |      1e-6      |      1e-4       |                      |                      |                |                |                 |
|  shopfacade   |          2           |          10          |       5        |      1e-4      |      1e-4       |          2           |          10          |       2        |      1e-6      |      1e-4       |
|  oldhospital  |         0.5          |          10          |       5        |      1e-6      |      1e-4       |          2           |          10          |       2        |      1e-5      |      1e-4       |
| stmaryschurch |          2           |          10          |       5        |      1e-6      |      1e-4       |                      |                      |                |                |                 |

Moreover, since our weight network takes only 5d pairs as input, it has better capacity of generalization than that relies on RGB input. One could use pretrained model from other scenes in `--w-model` as init to accelerate convergence. 

As for training epoch, since different scene has different size of training set, one could take an early stop according to the validation process, or simply replace `--epoch` with about 1.5M-2M iterations. 


##### End-to-end training:
`--lr` and `--glr` are set to 1e-5.

|     Scene     | loss_cla | loss_ess |  Ea   |  Eb   |
| :-----------: | :------: | :------: | :---: | :---: |
|     chess     |   0.5    |    10    |   5   | 1e-4  |
|     heads     |   0.5    |    10    |   1   | 1e-4  |
|     fire      |    2     |    10    |   2   | 1e-4  |
|    office     |   0.5    |    10    |   5   | 1e-4  |
|    pumpkin    |   0.5    |    10    |   5   | 1e-4  |
|    kitchen    |    2     |    10    |   2   | 1e-4  |
|    stairs     |   0.5    |    10    |   5   | 1e-4  |
|  greatcourt   |    8     |    10    |   2   | 1e-7  |
| kingscollege  |    2     |    10    |   5   | 1e-6  |
|  shopfacade   |    2     |    10    |   2   | 1e-4  |
|  oldhospital  |    2     |    10    |   2   | 1e-5  |
| stmaryschurch |    2     |    10    |   3   | 1e-6  |

As aforementioned, `--epoch` could be replaced with about 150k iterations.


### References
1. Dang, Zheng, et al. "Eigendecomposition-free training of deep networks for linear least-square problems." TPAMI (2020).

