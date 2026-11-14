| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm |
|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 193.8 | 195.2 | 194.2 | 195.7 | 195.1 | 196.1 | 196.1 | 197.7 |
| in_list_rank_change | 0.0 | -2.2 | -0.9 | -1.9 | -0.3 | -1.0 | 0.0 | 0.0 | 1.6 |
| in_list_moved_up | 0 | 248 | 248 | 249 | 247 | 235 | 368 | 177 | 186 |
| in_list_stayed | 514 | 45 | 34 | 43 | 40 | 49 | 53 | 54 | 35 |
| in_list_moved_down | 0 | 221 | 232 | 222 | 227 | 230 | 93 | 283 | 293 |
| in_list_top10 | 27 | 27 | 29 | 27 | 26 | 27 | 26 | 26 | 27 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 34 | 40 | 35 | 40 | 37 | 46 | 33 | 37 |
| all_mean_rank | 438.8 | 438.2 | 438.4 | 438.2 | 438.5 | 438.4 | 438.6 | 438.6 | 438.8 |

mean rank change of the in-list targets per user, bootstrap over users (95% interval):

| reranker | mean change | 2.5% | 97.5% |
|---|---|---|---|
| R1_like | -2.3 | -4.5 | -0.2 |
| R2_like-dis | -0.2 | -2.7 | +2.1 |
| R3_soft | -1.7 | -3.9 | +0.4 |
| R4_rocchio | +0.0 | -2.5 | +2.7 |
| R5_naive | -1.9 | -4.4 | +0.6 |
| R8_rev_like | +0.3 | -1.8 | +2.7 |
| R9_rev | +1.0 | -0.8 | +2.8 |
| R7_lgbm | +2.1 | -0.3 | +4.5 |

control: the same rerankers with another user's persona (what any persona moves):

| metric | lightgcn | R1_like | R2_like-dis | R3_soft | R4_rocchio | R5_naive | R8_rev_like | R9_rev | R7_lgbm |
|---|---|---|---|---|---|---|---|---|---|
| users | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| list_size | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 |
| targets | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 | 2518 |
| in_list | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 | 514 |
| in_list_mean_rank_lightgcn | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 | 196.1 |
| in_list_mean_rank | 196.1 | 195.2 | 196.5 | 195.1 | 197.2 | 195.7 | 196.0 | 196.8 | 198.0 |
| in_list_rank_change | 0.0 | -0.8 | 0.4 | -1.0 | 1.1 | -0.4 | -0.0 | 0.7 | 1.9 |
| in_list_moved_up | 0 | 236 | 239 | 237 | 230 | 221 | 347 | 160 | 170 |
| in_list_stayed | 514 | 38 | 38 | 38 | 41 | 45 | 72 | 55 | 30 |
| in_list_moved_down | 0 | 240 | 237 | 239 | 243 | 248 | 95 | 299 | 314 |
| in_list_top10 | 27 | 27 | 27 | 27 | 26 | 27 | 27 | 26 | 26 |
| out_of_list | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 | 2004 |
| out_pulled_in | 0 | 42 | 41 | 41 | 48 | 48 | 48 | 29 | 30 |
| all_mean_rank | 438.8 | 438.4 | 438.6 | 438.4 | 438.8 | 438.5 | 438.5 | 438.7 | 438.9 |

| reranker | in-list rank change, own persona | another user's persona |
|---|---|---|
| R1_like | -2.2 | -0.8 |
| R2_like-dis | -0.9 | +0.4 |
| R3_soft | -1.9 | -1.0 |
| R4_rocchio | -0.3 | +1.1 |
| R5_naive | -1.0 | -0.4 |
| R8_rev_like | +0.0 | -0.0 |
| R9_rev | +0.0 | +0.7 |
| R7_lgbm | +1.6 | +1.9 |
