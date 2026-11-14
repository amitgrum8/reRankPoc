run $dst | encoder BAAI/bge-small-en-v1.5 | alpha 1.0 beta 1.0 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 199.3 | 213.8 | 201.2 | 202.7 | 192.3 | 197.6 | 202.6 | 218.3 | 192.9 | 189.6 | 188.2 |
| in_list_rank_change | 0.0 | 3.2 | 17.7 | 5.2 | 6.7 | -3.8 | 1.5 | 6.6 | 22.3 | -3.1 | -6.5 | -7.9 |
| in_list_moved_up | 0 | 236 | 208 | 239 | 232 | 250 | 351 | 180 | 168 | 277 | 268 | 271 |
| in_list_stayed | 514 | 7 | 6 | 9 | 10 | 7 | 12 | 5 | 6 | 44 | 16 | 14 |
| in_list_moved_down | 0 | 271 | 300 | 266 | 272 | 257 | 151 | 329 | 340 | 193 | 230 | 229 |
| in_list_top10 | 27 | 27 | 29 | 25 | 27 | 30 | 25 | 23 | 26 | 26 | 28 | 32 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 825 | 822 | 818 | 891 | 924 | 739 | 764 | 712 | 33 | 149 | 438 |
| all_mean_rank | 438.8 | 423.3 | 427.5 | 424.7 | 424.1 | 420.8 | 424.2 | 426.2 | 425.9 | 438.0 | 436.4 | 431.3 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -4.4 | -17.7 | +9.4 |
| R2_like-dis | +16.1 | +4.4 | +28.2 |
| R3_soft | -1.1 | -14.2 | +12.4 |
| R4_rocchio | +5.2 | -7.9 | +18.1 |
| R5_naive | -13.7 | -27.4 | +0.2 |
| R8_rev_like | +3.6 | -7.2 | +16.1 |
| R9_rev | +9.4 | -0.6 | +19.2 |
| R7_lgbm | +24.4 | +13.9 | +34.3 |
| R10_mix | -4.2 | -6.1 | -2.2 |
| R11_mix_crit | -9.9 | -15.7 | -4.0 |
| R12_mix_reviews | -5.0 | -13.5 | +3.8 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 208.8 | 208.8 | 208.7 | 213.0 | 206.2 | 201.6 | 206.5 | 215.4 | 194.1 | 193.6 | 199.0 |
| in_list_rank_change | 0.0 | 12.7 | 12.7 | 12.6 | 17.0 | 10.1 | 5.6 | 10.4 | 19.4 | -2.0 | -2.5 | 2.9 |
| in_list_moved_up | 0 | 226 | 220 | 224 | 216 | 229 | 343 | 176 | 190 | 243 | 250 | 232 |
| in_list_stayed | 514 | 4 | 9 | 8 | 6 | 6 | 17 | 8 | 2 | 50 | 20 | 16 |
| in_list_moved_down | 0 | 284 | 285 | 282 | 292 | 279 | 154 | 330 | 322 | 221 | 244 | 266 |
| in_list_top10 | 27 | 25 | 29 | 27 | 23 | 24 | 24 | 21 | 29 | 26 | 30 | 28 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 815 | 827 | 812 | 905 | 875 | 744 | 747 | 647 | 37 | 148 | 435 |
| all_mean_rank | 438.8 | 426.4 | 427.5 | 427.1 | 427.0 | 426.5 | 424.9 | 426.5 | 427.5 | 438.3 | 437.2 | 433.6 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | +3.2 | +12.7 |
| R2_like-dis | +17.7 | +12.7 |
| R3_soft | +5.2 | +12.6 |
| R4_rocchio | +6.7 | +17.0 |
| R5_naive | -3.8 | +10.1 |
| R8_rev_like | +1.5 | +5.6 |
| R9_rev | +6.6 | +10.4 |
| R7_lgbm | +22.3 | +19.4 |
| R10_mix | -3.1 | -2.0 |
| R11_mix_crit | -6.5 | -2.5 |
| R12_mix_reviews | -7.9 | +2.9 |

R10_mix weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 10000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.315 | 0.025 |
| like_s1 | +0.006 | 0.010 |
| like_s2 | +0.007 | 0.010 |
| like_s3 | +0.005 | 0.010 |
| like_s4 | +0.002 | 0.010 |
| like_s5 | +0.003 | 0.010 |
| dis_s1 | +0.004 | 0.010 |
| dis_s2 | +0.000 | 0.010 |
| dis_s3 | +0.001 | 0.010 |
| dis_s4 | +0.002 | 0.010 |
| dis_s5 | +0.004 | 0.010 |
| sum of the like weights | +0.023 | 0.019 |
| sum of the dislike weights | +0.011 | 0.019 |

R11_mix_crit weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.312 | 0.025 |
| like_s1 | +0.026 | 0.025 |
| like_s2 | +0.029 | 0.027 |
| like_s3 | +0.014 | 0.027 |
| like_s4 | -0.004 | 0.027 |
| like_s5 | +0.006 | 0.026 |
| dis_crit_s1 | +0.032 | 0.026 |
| dis_crit_s2 | +0.020 | 0.027 |
| dis_crit_s3 | +0.006 | 0.028 |
| dis_crit_s4 | +0.005 | 0.027 |
| dis_crit_s5 | -0.024 | 0.027 |
| no_crit | -0.027 | 0.023 |
| sum of the like weights | +0.072 | 0.033 |
| sum of the dislike weights | +0.038 | 0.039 |

R12_mix_reviews weights: logistic regression on 413 training users with a target inside the list (944 targets, 206500 rows); inputs z-scored within each user's list; L2 penalty 1000, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.306 | 0.025 |
| like_good_s1 | +0.055 | 0.024 |
| like_good_s2 | +0.052 | 0.026 |
| like_good_s3 | +0.009 | 0.026 |
| like_good_s4 | +0.035 | 0.026 |
| like_good_s5 | +0.026 | 0.025 |
| dis_bad_s1 | +0.041 | 0.026 |
| dis_bad_s2 | +0.022 | 0.027 |
| dis_bad_s3 | +0.008 | 0.027 |
| dis_bad_s4 | +0.008 | 0.027 |
| dis_bad_s5 | +0.003 | 0.026 |
| no_bad | +0.005 | 0.024 |
| n_good | -0.014 | 0.023 |
| n_bad | -0.004 | 0.024 |
| sum of the like weights | +0.178 | 0.036 |
| sum of the dislike weights | +0.083 | 0.036 |
