## Whisper: Training Command Logs
```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--pooling_mode mean \
--loss cos_sim \
--num_epochs 50 \
--batch_size 512 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisper-384_mean_cos-sim_50_512_1e-5_1e-2 \
> logs/whisper-384_mean_cos-sim_50_512_1e-5_1e-2.txt
```
Done.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--pooling_mode mean \
--loss sim_clr \
--num_epochs 50 \
--batch_size 512 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisper-384_mean_sim-clr_50_512_1e-5_1e-2 \
> logs/whisper-384_mean_sim-clr_50_512_1e-5_1e-2.txt
```
Interrupted at Epoch 41.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--pooling_mode mean \
--loss norm_temp_ce_sum \
--num_epochs 50 \
--batch_size 512 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisper-384_mean_norm-temp-ce-sum_50_512_1e-5_1e-2 \
> logs/whisper-384_mean_norm-temp-ce-sum_50_512_1e-5_1e-2.txt
```
Done.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--use_sbert_encoder \
--pooling_mode mean \
--loss cos_sim \
--num_epochs 50 \
--batch_size 480 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisbert-384_mean_cos-sim_50_480_1e-5_1e-2 \
> logs/whisbert-384_mean_cos-sim_50_480_1e-5_1e-2.txt
```
Done.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--use_sbert_encoder \
--pooling_mode mean \
--loss norm_temp_ce_mean \
--num_epochs 50 \
--batch_size 480 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisbert-384_mean_norm-temp-ce-mean_50_480_1e-5_1e-2 \
> logs/whisbert-384_mean_norm-temp-ce-mean_50_480_1e-5_1e-2.txt
```
Done.


## WhisPA: Training Command Logs
```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--n_new_dims 13 \
--pooling_mode mean \
--loss cos_sim \
--num_epochs 50 \
--batch_size 600 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whispa-384_mean_cos-sim_50_600_1e-5_1e-2 \
> logs/whispa-384_mean_cos-sim_50_600_1e-5_1e-2.txt
```
Done.

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--n_new_dims 13 \
--pooling_mode mean \
--loss norm_temp_ce_sum \
--num_epochs 50 \
--batch_size 900 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whispa-384_mean_nce-sum_50_900_1e-5_1e-2 \
> logs/whispa-384_mean_nce-sum_50_900_1e-5_1e-2.txt
```
In Progress.

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--n_new_dims 13 \
--use_sbert_encoder \
--pooling_mode mean \
--loss cos_sim \
--num_epochs 50 \
--batch_size 500 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whispart-384_mean_cos-sim_50_500_1e-5_1e-2 \
> logs/whispart-384_mean_cos-sim_50_500_1e-5_1e-2.txt
```


##  768 EMBEDDING DIMENSIONS
```bash
CUDA_VISIBLE_DEVICES=2,3 \
python code/train.py \
--whisper_model_id openai/whisper-small \
--pooling_mode mean \
--loss cos_sim \
--num_epochs 50 \
--batch_size 100 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisper-768_mean_cos-sim_50_512_1e-5_1e-2 \
> logs/whisper-768_mean_cos-sim_50_512_1e-5_1e-2.txt
```
Interrupted at Epoch 29.
