# Neural Network Project: "A-ViT: Adaptive Tokens for Efficient Vision Transformer" (https://arxiv.org/pdf/2112.07658)

Students: Renato Giamba, Lorenzo Di Luccio

Matricola/Student IDs: 1816155, 1797569

Sapienza e-mail: giamba.1816155@studenti.uniroma1.it, diluccio.1797569@studenti.uniroma1.it

## Description of the paper
![alt text](https://a-vit.github.io/img_source/fig/teaser/teaser_web.png)

The authors of the paper introduced a new augmented vision transformer block.
With only two more fixed parameters (a weight and a bias, i.e. a single neuron), the augmented block is able to dynamically compute halting probability scores for each token starting from the classification one at the start, allowing to discard tokens that are uninformative.

The authors started from a pretrained DeiT transformer (we omitted the distillation part in our project for brevity, because it is not crucial for the final results) and then inserted the halting mechanism on top of that.

## A-ViT layer algorithm
![alt text](https://github.com/lorenzo-di-luccio/NNProject-tmp/blob/main/assets/images/alg2.PNG)

The modified version of the A-ViT layer algorithm, instead of simply computing the next hidden layer, perform some additional steps.

- Compute the next hidden states before the halting mechanism $\mathbf{t}=\text{F}(\mathbf{t} \odot \mathbf{m})$ using a token mask $\textbf{m}$ that will exclude previously halted tokens.
- Compute the halting probability scores $\mathbf{h}=\mathbf{\sigma}(\gamma \mathbf{t_{:,0}}+\mathbf{\beta})$ using the new fixed parameters $\beta$ and $\gamma$ or set all to ones (stop all) $\mathbf{h}=\mathbf{1}$ if it is the last layer.
- Update the cumulative halting score $\mathbf{cumul}=\mathbf{cumul}+\mathbf{h}$ and the ponder loss vector $\mathbf{\rho}=\mathbf{\rho}+\mathbf{m}$
- Update the reminder vector $\mathbf{R}$ and the ponder loss vector $\mathbf{\rho}$ for each token based on the fact that it has reached an halting probability score near 1 (if $\mathbf{cumul}_k \ge 1 - \epsilon$) or not (if $\mathbf{cumul}_k < 1 - \epsilon$).
- Update the next hidden states $\mathbf{out}$ for each token based on the fact that the classification token has reached an halting probability score near 1 (if $\mathbf{cumul}_c \ge 1 - \epsilon$)  or not (if $\mathbf{cumul}_c < 1 - \epsilon$).
- Update the mask $\mathbf{m}=\mathbf{cumul} < 1 - \epsilon$ so that it will exclude tokens that have reached an halting probability score near 1.

In the picture of the algorithm below, it is also showed the initial values and the relavant dimensions of the variables.

![alt text](https://github.com/lorenzo-di-luccio/NNProject-tmp/blob/main/assets/images/alg1.PNG)

## Losses
The total loss $\mathcal{L}$ is the sum of the task loss $\mathcal{L}\_{task}$ (classical cross-entropy loss for classification), the ponder loss $\mathcal{L}\_{ponder}$ (the average of the ponder vector, to encourage the tokens early stopping) and the distribution loss $\mathcal{L}\_{distr}$ (the KL divergence of the computed halting score probability distribution $\mathcal{H}$ and the target halting score probability distribution $\mathcal{H}\_{target}$, for regularization and to impose a preferred halting score probability distribution).

$$
\mathcal{H}\_{target}=\mathcal{N}(\mathbf{\mu}\_\mathcal{H}, \mathbf{I}) \\;\\;
\mathcal{H}=\frac{1}{K}
\begin{pmatrix}
\sum\limits_{k=1}^Kh_k^1 \\
\vdots \\
\sum\limits_{k=1}^Kh_k^L
\end{pmatrix}
$$

$$
\alpha_{ponder}=5 \cdot 10^{-4} \\;\\; \alpha_{distr}=0.1
$$

$$\mathcal{L}\_{ponder} = \frac{1}{K} \sum_{k=1}^K \rho_k \\;\\; \mathcal{L}\_{distr} = D_{KL}(H || H_{target})$$

$$\mathcal{L} = \mathcal{L}\_{task} + \alpha_{ponder}\mathcal{L}\_{ponder} + \alpha_{distr}\mathcal{L}\_{distr}$$

## Results

### Colab notebook
For more detailed results, plots, measures of inference time or for quickly running some visual tests, see [the dedicated Colab notebook](https://colab.research.google.com/drive/1sStCrRIWjoAiT1o3Myv0Q0WVARNFDYm0?usp=drive_link) for the project.

### Test metrics
|  | DeiT | AViT | AViT finetuned |
| --- | --- | --- | --- |
| **Accuracy** | 0.7322 | 0.6957 | 0.7121 |
| **F1 Score** | 0.7220 | 0.6873 | 0.7023 |

### Visual results
![alt text](https://github.com/lorenzo-di-luccio/NNProject-tmp/blob/main/assets/results/visual_test.png)

## Execution

### Creating the environment
Download the repository.
```bash
git clone https://github.com/lorenzo-di-luccio/NNProject.git
```

Create a Python virtual environment (recommended Python 3.10).
```bash
python -m venv <YOUR_ENV_DIR>
```

Activate the environment then move in the project (repository) directory.

Install all required packages.
```bash
pip install -r requirements.txt
```

### Downloading our trained models
If you want you can download our trained models and put them into thir respective subfolders (`AViT-tiny`, `AViT-tiny-finetuned`, `DeiT-tiny`) in the `assets` folder.

|  Model | Google Drive link |
| --- | --- |
| DeiT-tiny | https://drive.google.com/file/d/1vgAJYRbOq_ktCU3TH0v9syLCywC4wmp6/view?usp=drive_link |
| AViT-tiny | https://drive.google.com/file/d/1svBZjmoZZOSOHm21aB6Peau4wr8hKl1R/view?usp=drive_link |
| AViT-tiny (finetuned from our DeiT-tiny) | https://drive.google.com/file/d/1-0HhkfnuZhiNiD4cs9VqO2GuOAZHEN5w/view?usp=drive_link |

### Fitting
With the environment activated, move in the project (repository) directory.

Then run
```bash
python fit.py --batch_size <BATCH_SIZE> --model <MODEL> --lr <LR> --ckpt assets/<CKPT_PATH>.ckpt --device <DEVICE>
```
- `--batch_size`: The batch size for the dataset. Optional integer argument. Default is `128`.
- `--model`: The model to fit. Required string argument with choices between `["DeiT", "AViT"]`.
- `--lr`: The learning rate. Optional float argument. Default is `1.e-4`.
- `--ckpt`: The checkpoint to start with. Optional (relative) path argument. If not passed, fit from scratch.
- `--device`: The device to use. Optional string argument with choices between `["cpu", "gpu"]`. Default is `"cpu"`.

### Testing
With the environment activated, move in the project (repository) directory.

Then run
```bash
python test.py --batch_size <BATCH_SIZE> --model <MODEL> --ckpt assets/<CKPT_PATH>.ckpt --device <DEVICE>
```
- `--batch_size`: The batch size for the dataset. Optional integer argument. Default is `128`.
- `--model`: The model to test. Required argument with choices between `["DeiT", "AViT"]`.
- `--ckpt`: The checkpoint to load the weights from. Required (relative) path argument.
- `--device`: The device to use. Optional string argument with choices between `["cpu", "gpu"]`. Default is `"cpu"`.

### Running Visual Test
With the environment activated, move in the project (repository) directory.

Then run
```bash
python run_visual_test.py --ckpt assets/<CKPT_PATH>.ckpt --img_idxs <IMG_IDX_1> <IMG_IDX_2> ... <IMG_IDX_N>
```
- `--ckpt`: The AViT checkpoint to load the weights from. Required (relative) path argument.
- `--img_idxs`: The image indeces to run the test on. Optional integer list argument. Default is `[22, 29, 40, 43, 76, 109]`
