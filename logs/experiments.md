## WhiSA: Training Command Logs
```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--pooling_mode mean \
--loss CS \
--num_epochs 50 \
--batch_size 900 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whisper-384_cs_50_900_1e-5_1e-2 \
> logs/whisper-384_cs_50_900_1e-5_1e-2.txt \
&& CUDA_VISIBLE_DEVICES=1,2,3 \
python code/inference.py \
--load_name whisper-384_cs_50_900_1e-5_1e-2 \
--batch_size 3096 \
--num_workers 16 \
--no_shuffle
```
Done.

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


## WhiSPA: Training Command Logs
```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--n_new_dims 10 \
--pooling_mode mean \
--loss NCE \
--num_epochs 50 \
--batch_size 900 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whispa-384_nce_50_900_1e-5_1e-2 \
> logs/whispa-384_nce_50_900_1e-5_1e-2.txt \
&& CUDA_VISIBLE_DEVICES=1,2,3 \
python code/inference.py \
--load_name whispa-384_nce_50_900_1e-5_1e-2 \
--batch_size 3096 \
--num_workers 16 \
--no_shuffle
```
- Note that cs ran -10 SBERT feats
Done. 

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
python code/train.py \
--whisper_model_id openai/whisper-tiny \
--n_new_dims 10 \
--pooling_mode mean \
--loss norm_temp_ce_sum \
--num_epochs 50 \
--batch_size 900 \
--num_workers 16 \
--lr 1e-5 \
--wd 1e-2 \
--save_name whispa-384_mean_nce-sum_50_900_1e-5_1e-2 \
> logs/whispa-384_mean_nce-sum_50_900_1e-5_1e-2.txt \
&& CUDA_VISIBLE_DEVICES=1,2,3 \
python code/inference.py \
--load_name whispa-384_mean_nce-sum_50_900_1e-5_1e-2 \
--batch_size 3096 \
--num_workers 16 \
--no_shuffle
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
