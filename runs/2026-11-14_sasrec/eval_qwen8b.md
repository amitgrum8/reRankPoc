run 2026-11-14_sasrec | encoder Qwen/Qwen3-Embedding-8B | alpha 0.1 beta 0.1 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 |
| in_list_mean_rank_lightgcn | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 |
| in_list_mean_rank | 167.5 | 163.8 | 166.5 | 164.5 | 165.1 | 163.0 | 166.6 | 166.2 | 170.8 | 157.0 | 156.2 | 157.8 |
| in_list_rank_change | 0.0 | -3.7 | -1.0 | -2.9 | -2.4 | -4.5 | -0.9 | -1.2 | 3.3 | -10.5 | -11.3 | -9.7 |
| in_list_moved_up | 0 | 288 | 258 | 285 | 277 | 302 | 342 | 207 | 192 | 300 | 301 | 304 |
| in_list_stayed | 543 | 42 | 47 | 45 | 46 | 47 | 64 | 58 | 53 | 18 | 22 | 16 |
| in_list_moved_down | 0 | 213 | 238 | 213 | 220 | 194 | 137 | 278 | 298 | 225 | 220 | 223 |
| in_list_top10 | 41 | 41 | 46 | 41 | 46 | 41 | 40 | 44 | 44 | 47 | 48 | 50 |
| out_of_list | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 |
| out_pulled_in | 0 | 31 | 36 | 30 | 40 | 38 | 42 | 28 | 22 | 264 | 286 | 286 |
| all_mean_rank | 429.1 | 428.1 | 428.7 | 428.3 | 428.4 | 427.9 | 428.7 | 428.6 | 429.7 | 422.7 | 422.6 | 422.7 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -4.4 | -6.6 | -2.3 |
| R2_like-dis | -1.9 | -4.3 | +0.3 |
| R3_soft | -3.6 | -5.8 | -1.6 |
| R4_rocchio | -2.4 | -4.5 | -0.5 |
| R5_naive | -4.7 | -6.7 | -2.7 |
| R8_rev_like | -0.2 | -2.1 | +2.0 |
| R9_rev | -0.4 | -2.3 | +1.4 |
| R7_lgbm | +2.6 | +0.7 | +4.5 |
| R10_mix | -12.7 | -20.3 | -5.2 |
| R11_mix_crit | -14.7 | -22.1 | -7.5 |
| R12_mix_reviews | -11.2 | -17.6 | -5.1 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 | 543 |
| in_list_mean_rank_lightgcn | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 | 167.5 |
| in_list_mean_rank | 167.5 | 168.7 | 167.8 | 168.5 | 168.4 | 168.3 | 168.0 | 168.0 | 168.3 | 174.5 | 172.9 | 171.2 |
| in_list_rank_change | 0.0 | 1.2 | 0.4 | 1.0 | 0.9 | 0.8 | 0.5 | 0.5 | 0.8 | 7.0 | 5.4 | 3.7 |
| in_list_moved_up | 0 | 209 | 254 | 211 | 243 | 226 | 336 | 185 | 187 | 217 | 233 | 249 |
| in_list_stayed | 543 | 50 | 53 | 50 | 49 | 52 | 66 | 61 | 59 | 19 | 16 | 17 |
| in_list_moved_down | 0 | 284 | 236 | 282 | 251 | 265 | 141 | 297 | 297 | 307 | 294 | 277 |
| in_list_top10 | 41 | 40 | 43 | 40 | 42 | 40 | 39 | 41 | 41 | 36 | 36 | 37 |
| out_of_list | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 | 1975 |
| out_pulled_in | 0 | 29 | 40 | 29 | 37 | 30 | 38 | 31 | 27 | 250 | 237 | 282 |
| all_mean_rank | 429.1 | 429.2 | 429.0 | 429.2 | 429.1 | 429.1 | 429.1 | 429.1 | 429.1 | 427.9 | 427.5 | 427.1 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -3.7 | +1.2 |
| R2_like-dis | -1.0 | +0.4 |
| R3_soft | -2.9 | +1.0 |
| R4_rocchio | -2.4 | +0.9 |
| R5_naive | -4.5 | +0.8 |
| R8_rev_like | -0.9 | +0.5 |
| R9_rev | -1.2 | +0.5 |
| R7_lgbm | +3.3 | +0.8 |
| R10_mix | -10.5 | +7.0 |
| R11_mix_crit | -11.3 | +5.4 |
| R12_mix_reviews | -9.7 | +3.7 |

R10_mix weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.448 | 0.023 |
| like_s1 | +0.059 | 0.025 |
| like_s2 | +0.060 | 0.026 |
| like_s3 | +0.038 | 0.027 |
| like_s4 | +0.021 | 0.026 |
| like_s5 | +0.022 | 0.025 |
| dis_s1 | +0.028 | 0.025 |
| dis_s2 | +0.020 | 0.027 |
| dis_s3 | +0.026 | 0.027 |
| dis_s4 | +0.002 | 0.027 |
| dis_s5 | -0.033 | 0.026 |
| sum of the like weights | +0.200 | 0.036 |
| sum of the dislike weights | +0.043 | 0.036 |

R11_mix_crit weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.448 | 0.023 |
| like_s1 | +0.064 | 0.025 |
| like_s2 | +0.064 | 0.026 |
| like_s3 | +0.041 | 0.026 |
| like_s4 | +0.024 | 0.026 |
| like_s5 | +0.024 | 0.025 |
| dis_crit_s1 | +0.015 | 0.026 |
| dis_crit_s2 | +0.018 | 0.027 |
| dis_crit_s3 | +0.020 | 0.028 |
| dis_crit_s4 | +0.022 | 0.027 |
| dis_crit_s5 | +0.000 | 0.027 |
| no_crit | -0.015 | 0.022 |
| sum of the like weights | +0.217 | 0.033 |
| sum of the dislike weights | +0.074 | 0.038 |

R12_mix_reviews weights: logistic regression on 414 training users with a target inside the list (1019 targets, 207000 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.444 | 0.023 |
| like_good_s1 | +0.045 | 0.024 |
| like_good_s2 | +0.043 | 0.026 |
| like_good_s3 | +0.045 | 0.026 |
| like_good_s4 | +0.043 | 0.026 |
| like_good_s5 | +0.000 | 0.025 |
| dis_bad_s1 | +0.034 | 0.025 |
| dis_bad_s2 | +0.037 | 0.027 |
| dis_bad_s3 | +0.015 | 0.027 |
| dis_bad_s4 | +0.015 | 0.027 |
| dis_bad_s5 | +0.025 | 0.026 |
| no_bad | -0.030 | 0.025 |
| n_good | -0.009 | 0.024 |
| n_bad | +0.048 | 0.024 |
| sum of the like weights | +0.176 | 0.034 |
| sum of the dislike weights | +0.127 | 0.037 |
