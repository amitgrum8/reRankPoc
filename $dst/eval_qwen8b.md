run $dst | encoder Qwen/Qwen3-Embedding-8B | alpha 0.1 beta 0.1 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 192.5 | 195.4 | 193.3 | 193.1 | 190.8 | 194.0 | 195.1 | 199.3 | 183.9 | 183.9 | 177.2 |
| in_list_rank_change | 0.0 | -3.6 | -0.7 | -2.8 | -3.0 | -5.3 | -2.0 | -0.9 | 3.3 | -12.1 | -12.2 | -18.9 |
| in_list_moved_up | 0 | 265 | 238 | 261 | 262 | 294 | 330 | 198 | 176 | 277 | 291 | 285 |
| in_list_stayed | 514 | 49 | 39 | 46 | 43 | 38 | 57 | 49 | 43 | 8 | 13 | 13 |
| in_list_moved_down | 0 | 200 | 237 | 207 | 209 | 182 | 127 | 267 | 295 | 229 | 210 | 216 |
| in_list_top10 | 27 | 29 | 26 | 29 | 26 | 28 | 27 | 27 | 26 | 31 | 31 | 31 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 32 | 41 | 30 | 39 | 36 | 47 | 37 | 47 | 384 | 286 | 621 |
| all_mean_rank | 438.8 | 437.8 | 438.4 | 438.0 | 437.9 | 437.4 | 438.1 | 438.4 | 439.2 | 429.8 | 433.1 | 423.4 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -4.0 | -6.4 | -1.6 |
| R2_like-dis | -0.3 | -3.2 | +2.7 |
| R3_soft | -3.0 | -5.6 | -0.5 |
| R4_rocchio | -2.6 | -5.4 | +0.1 |
| R5_naive | -5.6 | -7.9 | -3.3 |
| R8_rev_like | -1.5 | -3.8 | +0.9 |
| R9_rev | -0.0 | -2.0 | +1.9 |
| R7_lgbm | +3.8 | +1.6 | +5.8 |
| R10_mix | -16.8 | -26.5 | -6.9 |
| R11_mix_crit | -15.9 | -23.9 | -7.7 |
| R12_mix_reviews | -17.3 | -27.8 | -6.4 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 195.1 | 194.6 | 195.3 | 194.7 | 195.2 | 195.4 | 196.2 | 198.7 | 201.6 | 196.0 | 202.4 |
| in_list_rank_change | 0.0 | -1.0 | -1.5 | -0.8 | -1.3 | -0.9 | -0.6 | 0.1 | 2.7 | 5.6 | -0.0 | 6.3 |
| in_list_moved_up | 0 | 232 | 255 | 230 | 244 | 241 | 323 | 189 | 182 | 221 | 237 | 236 |
| in_list_stayed | 514 | 43 | 49 | 45 | 43 | 36 | 65 | 51 | 47 | 6 | 11 | 6 |
| in_list_moved_down | 0 | 239 | 210 | 239 | 227 | 237 | 126 | 274 | 285 | 287 | 266 | 272 |
| in_list_top10 | 27 | 26 | 26 | 26 | 26 | 27 | 28 | 27 | 29 | 28 | 29 | 24 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 32 | 42 | 32 | 42 | 40 | 50 | 32 | 58 | 371 | 271 | 595 |
| all_mean_rank | 438.8 | 438.4 | 438.3 | 438.5 | 438.3 | 438.4 | 438.4 | 438.6 | 439.1 | 434.4 | 436.0 | 429.6 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -3.6 | -1.0 |
| R2_like-dis | -0.7 | -1.5 |
| R3_soft | -2.8 | -0.8 |
| R4_rocchio | -3.0 | -1.3 |
| R5_naive | -5.3 | -0.9 |
| R8_rev_like | -2.0 | -0.6 |
| R9_rev | -0.9 | +0.1 |
| R7_lgbm | +3.3 | +2.7 |
| R10_mix | -12.1 | +5.6 |
| R11_mix_crit | -12.2 | -0.0 |
| R12_mix_reviews | -18.9 | +6.3 |

R10_mix weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.301 | 0.025 |
| like_s1 | +0.070 | 0.042 |
| like_s2 | +0.060 | 0.051 |
| like_s3 | +0.044 | 0.054 |
| like_s4 | +0.001 | 0.052 |
| like_s5 | -0.021 | 0.044 |
| dis_s1 | +0.113 | 0.044 |
| dis_s2 | -0.069 | 0.056 |
| dis_s3 | +0.068 | 0.059 |
| dis_s4 | -0.004 | 0.057 |
| dis_s5 | -0.096 | 0.047 |
| sum of the like weights | +0.155 | 0.045 |
| sum of the dislike weights | +0.012 | 0.045 |

R11_mix_crit weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.305 | 0.025 |
| like_s1 | +0.050 | 0.025 |
| like_s2 | +0.042 | 0.026 |
| like_s3 | +0.030 | 0.026 |
| like_s4 | +0.011 | 0.026 |
| like_s5 | -0.004 | 0.025 |
| dis_crit_s1 | +0.006 | 0.026 |
| dis_crit_s2 | +0.013 | 0.027 |
| dis_crit_s3 | +0.031 | 0.028 |
| dis_crit_s4 | +0.013 | 0.027 |
| dis_crit_s5 | -0.008 | 0.027 |
| no_crit | -0.028 | 0.023 |
| sum of the like weights | +0.129 | 0.034 |
| sum of the dislike weights | +0.055 | 0.039 |

R12_mix_reviews weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.305 | 0.025 |
| like_good_s1 | +0.118 | 0.039 |
| like_good_s2 | -0.013 | 0.048 |
| like_good_s3 | +0.065 | 0.051 |
| like_good_s4 | +0.103 | 0.049 |
| like_good_s5 | -0.056 | 0.040 |
| dis_bad_s1 | +0.106 | 0.042 |
| dis_bad_s2 | +0.027 | 0.052 |
| dis_bad_s3 | -0.022 | 0.057 |
| dis_bad_s4 | +0.015 | 0.056 |
| dis_bad_s5 | +0.010 | 0.048 |
| no_bad | +0.012 | 0.037 |
| n_good | -0.034 | 0.034 |
| n_bad | -0.011 | 0.039 |
| sum of the like weights | +0.218 | 0.041 |
| sum of the dislike weights | +0.135 | 0.044 |
