run 2026-11-14_pop | encoder Qwen/Qwen3-Embedding-8B | alpha 0.1 beta 0.1 gamma 1.0

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 |
| in_list_mean_rank_lightgcn | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 |
| in_list_mean_rank | 228.1 | 218.4 | 222.9 | 219.2 | 223.0 | 216.2 | 228.0 | 227.2 | 215.7 | 186.1 | 181.1 | 176.6 |
| in_list_rank_change | 0.0 | -9.7 | -5.2 | -8.9 | -5.1 | -11.9 | -0.1 | -0.9 | -12.4 | -42.0 | -47.0 | -51.5 |
| in_list_moved_up | 0 | 115 | 105 | 114 | 101 | 128 | 126 | 81 | 109 | 119 | 119 | 119 |
| in_list_stayed | 189 | 9 | 7 | 5 | 6 | 9 | 9 | 10 | 7 | 1 | 3 | 0 |
| in_list_moved_down | 0 | 65 | 77 | 70 | 82 | 52 | 54 | 98 | 73 | 69 | 67 | 70 |
| in_list_top10 | 5 | 5 | 3 | 4 | 3 | 4 | 5 | 5 | 4 | 6 | 7 | 6 |
| out_of_list | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 |
| out_pulled_in | 0 | 35 | 40 | 35 | 34 | 36 | 33 | 19 | 26 | 991 | 1055 | 1520 |
| all_mean_rank | 480.5 | 479.5 | 479.8 | 479.5 | 479.9 | 479.3 | 480.3 | 480.3 | 479.2 | 447.6 | 446.3 | 432.1 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -9.8 | -14.2 | -5.6 |
| R2_like-dis | -5.6 | -10.8 | -0.7 |
| R3_soft | -8.9 | -13.5 | -4.7 |
| R4_rocchio | -4.7 | -8.9 | -0.8 |
| R5_naive | -11.8 | -15.5 | -8.1 |
| R8_rev_like | +0.7 | -3.1 | +4.7 |
| R9_rev | -0.4 | -3.7 | +2.8 |
| R7_lgbm | -11.9 | -16.3 | -7.8 |
| R10_mix | -41.4 | -62.6 | -21.1 |
| R11_mix_crit | -49.1 | -71.4 | -29.0 |
| R12_mix_reviews | -58.3 | -79.6 | -37.3 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm | R10_mix | R11_mix_crit | R12_mix_reviews |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 | 189 |
| in_list_mean_rank_lightgcn | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 | 228.1 |
| in_list_mean_rank | 228.1 | 230.9 | 234.2 | 231.7 | 231.5 | 228.9 | 228.6 | 227.9 | 227.8 | 245.5 | 242.0 | 221.3 |
| in_list_rank_change | 0.0 | 2.8 | 6.1 | 3.6 | 3.4 | 0.8 | 0.5 | -0.2 | -0.3 | 17.4 | 13.9 | -6.8 |
| in_list_moved_up | 0 | 77 | 80 | 77 | 84 | 83 | 120 | 73 | 65 | 80 | 76 | 91 |
| in_list_stayed | 189 | 7 | 9 | 11 | 8 | 6 | 10 | 10 | 8 | 1 | 1 | 0 |
| in_list_moved_down | 0 | 105 | 100 | 101 | 97 | 100 | 59 | 106 | 116 | 108 | 112 | 98 |
| in_list_top10 | 5 | 6 | 4 | 6 | 4 | 5 | 6 | 6 | 6 | 9 | 6 | 8 |
| out_of_list | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 | 2329 |
| out_pulled_in | 0 | 22 | 29 | 23 | 26 | 21 | 31 | 19 | 12 | 690 | 760 | 1209 |
| all_mean_rank | 480.5 | 480.6 | 480.8 | 480.7 | 480.7 | 480.5 | 480.4 | 480.4 | 480.4 | 469.4 | 468.3 | 456.8 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -9.7 | +2.8 |
| R2_like-dis | -5.2 | +6.1 |
| R3_soft | -8.9 | +3.6 |
| R4_rocchio | -5.1 | +3.4 |
| R5_naive | -11.9 | +0.8 |
| R8_rev_like | -0.1 | +0.5 |
| R9_rev | -0.9 | -0.2 |
| R7_lgbm | -12.4 | -0.3 |
| R10_mix | -42.0 | +17.4 |
| R11_mix_crit | -47.0 | +13.9 |
| R12_mix_reviews | -51.5 | -6.8 |

R10_mix weights: logistic regression on 265 training users with a target inside the list (479 targets, 132500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.323 | 0.038 |
| like_s1 | +0.206 | 0.051 |
| like_s2 | +0.029 | 0.063 |
| like_s3 | +0.026 | 0.067 |
| like_s4 | +0.076 | 0.065 |
| like_s5 | +0.056 | 0.056 |
| dis_s1 | +0.041 | 0.056 |
| dis_s2 | -0.037 | 0.068 |
| dis_s3 | -0.002 | 0.070 |
| dis_s4 | +0.030 | 0.069 |
| dis_s5 | -0.061 | 0.059 |
| sum of the like weights | +0.394 | 0.060 |
| sum of the dislike weights | -0.029 | 0.059 |

R11_mix_crit weights: logistic regression on 265 training users with a target inside the list (479 targets, 132500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.319 | 0.039 |
| like_s1 | +0.207 | 0.051 |
| like_s2 | +0.026 | 0.063 |
| like_s3 | +0.023 | 0.067 |
| like_s4 | +0.072 | 0.065 |
| like_s5 | +0.049 | 0.055 |
| dis_crit_s1 | -0.061 | 0.060 |
| dis_crit_s2 | +0.114 | 0.068 |
| dis_crit_s3 | +0.051 | 0.070 |
| dis_crit_s4 | +0.022 | 0.069 |
| dis_crit_s5 | -0.098 | 0.062 |
| no_crit | +0.001 | 0.042 |
| sum of the like weights | +0.377 | 0.052 |
| sum of the dislike weights | +0.028 | 0.063 |

R12_mix_reviews weights: logistic regression on 265 training users with a target inside the list (479 targets, 132500 rows); inputs z-scored within each user's list; L2 penalty 100, picked by 5-fold cross-validation over users from 1, 10, 100, 1000, 10000, 100000:

| input | weight | std. error |
|---|---|---|
| base | +0.292 | 0.040 |
| like_good_s1 | +0.092 | 0.051 |
| like_good_s2 | +0.033 | 0.061 |
| like_good_s3 | +0.157 | 0.064 |
| like_good_s4 | +0.087 | 0.061 |
| like_good_s5 | +0.041 | 0.053 |
| dis_bad_s1 | +0.152 | 0.052 |
| dis_bad_s2 | +0.005 | 0.063 |
| dis_bad_s3 | -0.043 | 0.068 |
| dis_bad_s4 | +0.086 | 0.066 |
| dis_bad_s5 | +0.057 | 0.059 |
| no_bad | -0.050 | 0.056 |
| n_good | -0.097 | 0.045 |
| n_bad | +0.009 | 0.048 |
| sum of the like weights | +0.410 | 0.054 |
| sum of the dislike weights | +0.257 | 0.059 |
